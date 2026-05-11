# heartbit-ghost P1.7 — Loop & cost protection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-10-heartbit-ghost-p1.7-loop-protection-design.md`

**Goal:** Prevent AI-to-AI mention loops and cap LLM cost on the existing P1.5 reactive replies pipeline. Adds four orthogonal guards (thread-depth, bot-heuristic, conversation-depth, daily-budget) that short-circuit cheaply before the reply pipeline is invoked.

**Architecture:** Mirror exact of P1.5's existing `SpamGuard` pattern. New value-type fields on `Mention` (`conversation_id`) and `MentionerContext` (`following_count`, `account_created_at`); `MentionStore` gains 2 conversation-counter methods; new `DailyTokenBudget` trait + 2 impls. Each guard is a small standalone struct with a `should_skip(...) -> Option<SkipReason>` method. The handler chains them in order before dispatching `ReplyDraft`.

**Tech Stack:** Rust 2024 edition, tokio, serde, chrono. Existing crates: `heartbit-core`, `heartbit-ghost`, `heartbit`. Reuses `MentionStore` (P1.5 task 7), `Mention`/`MentionerContext` (P1.5 task 2), `handle_mention_poll` (P1.5 task 10), `TokenUsage` (heartbit-core).

**Branch:** `feat/heartbit-ghost-p1.7-loop-protection` (already created off main; spec already committed there).

**Sub-phases (per spec §13):**
- **P1.7a** — Value-type extensions + `ThreadDepthGuard` (Tasks 1-5)
- **P1.7b** — `BotHeuristicGuard` + `MentionStore` extensions + `ConversationDepthGuard` (Tasks 6-8)
- **P1.7c** — `DailyTokenBudget` trait + 2 impls + `DailyBudgetGuard` (Tasks 9-11)
- **P1.7d** — Integration into `handle_mention_poll` + acceptance (Tasks 12-13)

---

## Task 1: Extend `Mention` struct with `conversation_id`

**Files:**
- Modify: `crates/heartbit-ghost/src/reply/mod.rs` (add field to `Mention` struct)

- [ ] **Step 1: Add the field**

In `crates/heartbit-ghost/src/reply/mod.rs`, in `pub struct Mention { ... }` (around line 18), add a new field after `in_reply_to_tweet_id`:

```rust
    /// X conversation_id (the root tweet of the thread tree). Used by
    /// the conversation-depth guard (P1.7) to cap reply count per
    /// conversation. `#[serde(default)]` for backward compatibility
    /// with stores written before P1.7.
    #[serde(default)]
    pub conversation_id: Option<String>,
```

Final struct:
```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Mention {
    pub id: String,
    pub text: String,
    pub author_id: String,
    pub author_handle: String,
    pub posted_at: DateTime<Utc>,
    pub in_reply_to_tweet_id: Option<String>,
    #[serde(default)]
    pub conversation_id: Option<String>,
}
```

- [ ] **Step 2: Update existing tests + fixtures**

```bash
grep -rn "Mention {" crates/ --include="*.rs"
```

Every `Mention { ... }` literal needs `conversation_id: None` (or a real value where appropriate). Most are in test fixtures. Use `cargo build` to find them; the compiler errors will guide.

- [ ] **Step 3: Add a test for the new field's serde default**

Append to `crates/heartbit-ghost/src/reply/mod.rs::tests`:

```rust
    #[test]
    fn mention_deserializes_without_conversation_id_field() {
        // Backward compat: old stores wrote Mention without the field.
        let json = r#"{
            "id": "1",
            "text": "hi",
            "author_id": "12",
            "author_handle": "alice",
            "posted_at": "2026-05-08T11:02:00Z",
            "in_reply_to_tweet_id": null
        }"#;
        let m: Mention = serde_json::from_str(json).expect("backward compat");
        assert!(m.conversation_id.is_none());
    }

    #[test]
    fn mention_round_trips_conversation_id() {
        let m = Mention {
            id: "1".into(),
            text: "hi".into(),
            author_id: "12".into(),
            author_handle: "alice".into(),
            posted_at: Utc::now(),
            in_reply_to_tweet_id: Some("99".into()),
            conversation_id: Some("conv-123".into()),
        };
        let s = serde_json::to_string(&m).unwrap();
        let parsed: Mention = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.conversation_id.as_deref(), Some("conv-123"));
    }
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p heartbit-ghost --lib reply
cargo build --workspace
```

Expected: build clean, all reply tests pass (including the 2 new ones).

- [ ] **Step 5: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add crates/heartbit-ghost/src/reply/mod.rs $(git diff --name-only | tr '\n' ' ')
git commit -m "$(cat <<'EOF'
feat(ghost): Mention.conversation_id — plumbing for P1.7 conversation-depth guard

Adds Option<String> field with #[serde(default)] for backward compat
with stores written before P1.7. Two new tests cover serde default
and round-trip. All existing fixtures updated to set conversation_id: None.

heartbit-ghost P1.7a — task 1/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Extend `MentionerContext` with `following_count` + `account_created_at`

**Files:**
- Modify: `crates/heartbit-ghost/src/reply/mod.rs` (add 2 fields to `MentionerContext`)

- [ ] **Step 1: Add the fields**

In `crates/heartbit-ghost/src/reply/mod.rs::MentionerContext` (around line 49), after `follower_count`:

```rust
    /// Following count of the mentioner, if available. Used by the
    /// bot-heuristic guard (P1.7) for the follower/following ratio
    /// signal.
    pub following_count: Option<u64>,
    /// When the mentioner's account was created. Used by the
    /// bot-heuristic guard (P1.7) for the account-age signal.
    pub account_created_at: Option<DateTime<Utc>>,
```

Final struct:
```rust
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MentionerContext {
    pub handle: String,
    pub bio: Option<String>,
    pub recent_tweets: Vec<TweetSnapshot>,
    pub follower_count: Option<u64>,
    pub following_count: Option<u64>,
    pub account_created_at: Option<DateTime<Utc>>,
}
```

`Default` derive picks up `None` for both new fields automatically — no impl change needed.

- [ ] **Step 2: Run cargo build**

```bash
cargo build --workspace
```

Expected: clean (the new fields are `Option<...>` with `Default::default() = None`, so existing literals are unchanged. Any literal using `..MentionerContext::default()` keeps working).

- [ ] **Step 3: Find any explicit literals that need updating**

```bash
grep -rn "MentionerContext {" crates/ --include="*.rs"
```

For each literal, ensure it either uses `..MentionerContext::default()` or sets the new fields explicitly. The default approach is preferred (safer for future field additions).

- [ ] **Step 4: Add a test**

Append to `crates/heartbit-ghost/src/reply/mod.rs::tests`:

```rust
    #[test]
    fn mentioner_context_default_has_none_for_new_fields() {
        let m = MentionerContext::default();
        assert!(m.following_count.is_none());
        assert!(m.account_created_at.is_none());
    }
```

- [ ] **Step 5: Run + format + commit**

```bash
cargo test -p heartbit-ghost --lib reply
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add crates/heartbit-ghost/src/reply/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): MentionerContext.{following_count, account_created_at}

Two new optional fields for the bot-heuristic guard (P1.7).
following_count enables the follower/following ratio signal;
account_created_at enables the account-age signal. Both default to
None via the derived Default impl.

heartbit-ghost P1.7a — task 2/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Extend `TwitterMentionsTool` to fetch `conversation_id`

**Files:**
- Modify: `crates/heartbit-ghost/src/tools/mentions.rs` (add `conversation_id` to query + response struct)

- [ ] **Step 1: Read the existing file**

```bash
sed -n '1,80p' crates/heartbit-ghost/src/tools/mentions.rs
```

Find the `MentionsApiResponse` / item struct (the deserialization shape) and the `tweet.fields` query string in `call_x`.

- [ ] **Step 2: Add `conversation_id` to the response struct**

The X API mentions response has tweet objects with optional fields. The internal item struct (likely named `MentionItem` or similar) needs a new field:

```rust
    #[serde(default)]
    pub conversation_id: Option<String>,
```

Find the existing struct (search for `pub struct Mention` or similar inside this file — note this is a private/internal struct, NOT the public `crate::reply::Mention`). Add the field.

- [ ] **Step 3: Update the query**

In `call_x` (or wherever the GET path is built), the existing `tweet.fields` query has `author_id,created_at,in_reply_to_user_id`. Add `conversation_id`:

```rust
("tweet.fields", "author_id,created_at,in_reply_to_user_id,conversation_id"),
```

- [ ] **Step 4: Map into the public `Mention`**

Find where the tool maps the API response into a public `crate::reply::Mention` (or a JSON structure consumed by the handler). Set `conversation_id` from the API item:

```rust
crate::reply::Mention {
    // ... existing fields ...
    conversation_id: item.conversation_id.clone(),
}
```

If the tool returns JSON (not a `Mention` directly), update the JSON serialization path to include `conversation_id`. Look at how `in_reply_to_tweet_id` (or `in_reply_to_user_id`) is mapped — mirror that.

- [ ] **Step 5: Add a wiremock test for the new field**

Append to `crates/heartbit-ghost/src/tools/mentions.rs::tests`:

```rust
    #[tokio::test]
    async fn mentions_includes_conversation_id_when_field_requested() {
        use wiremock::matchers::{method, path as wm_path, query_param_contains};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100/mentions"))
            .and(query_param_contains("tweet.fields", "conversation_id"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {
                        "id": "9001",
                        "text": "hey @user",
                        "author_id": "200",
                        "created_at": "2026-01-01T00:00:00.000Z",
                        "in_reply_to_user_id": "100",
                        "conversation_id": "conv-root-1"
                    }
                ],
                "meta": {}
            })))
            .mount(&server)
            .await;
        let client = test_client(&server.uri());
        let input = MentionsInput {
            user_id: "100".into(),
            max_results: 10,
            since_id: None,
        };
        let result = call_x(&client, &input).await.expect("happy path");
        // Adapt this assertion to whatever the tool returns. If it returns
        // an internal MentionsApiResponse struct, assert the conversation_id
        // is on the first element.
        // If the tool returns JSON, deserialize and check.
    }
```

The exact assertion shape depends on how `call_x` returns its result — adjust to match. The point is: with `conversation_id` in the fields query, the response containing `"conversation_id": "conv-root-1"` should successfully parse and surface the value to the caller.

If `query_param_contains` is not available in the wiremock version used, fall back to `query_param("tweet.fields", "author_id,created_at,in_reply_to_user_id,conversation_id")` (full string match).

- [ ] **Step 6: Run + format + commit**

```bash
cargo test -p heartbit-ghost --lib tools::mentions
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add crates/heartbit-ghost/src/tools/mentions.rs
git commit -m "$(cat <<'EOF'
feat(ghost): TwitterMentionsTool fetches conversation_id

Adds conversation_id to the tweet.fields query and to the response
struct so the public Mention struct (P1.7 task 1) gets populated.
One wiremock test asserts the field is requested and parsed.

heartbit-ghost P1.7a — task 3/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Extend `SkipReason` enum with P1.7 variants

**Files:**
- Modify: `crates/heartbit-ghost/src/reply/spam_guard.rs` (extend enum + tests)

- [ ] **Step 1: Read the existing enum**

```bash
sed -n '8,30p' crates/heartbit-ghost/src/reply/spam_guard.rs
```

Confirm the existing 5 variants: `SelfReply`, `StaleParent`, `LowEffortSpam`, `PerAuthorRateLimit`, `TooShortToEngage`.

- [ ] **Step 2: Add 4 new variants**

In `crates/heartbit-ghost/src/reply/spam_guard.rs`, in `pub enum SkipReason { ... }`:

```rust
    /// P1.7: this mention's parent tweet is in our replied set —
    /// continuing the loop would re-engage with our own thread.
    OwnThreadContinuation,
    /// P1.7: ≥2 of 3 bot-heuristic signals matched.
    BotSuspected {
        /// Human-readable reasons (e.g. "handle suffix '_bot'", "follower/following ratio 0.02 < 0.05").
        reasons: Vec<String>,
    },
    /// P1.7: this conversation already has at least `cap` replies from us.
    ConversationDepthExceeded {
        /// X conversation_id (the root tweet of the thread tree).
        conversation_id: String,
        /// Number of replies we've already sent in this conversation.
        count: usize,
        /// The configured cap.
        cap: usize,
    },
    /// P1.7: persona's daily LLM token budget is exhausted for today.
    DailyBudgetExhausted {
        /// Tokens already used today.
        used: u64,
        /// Configured budget.
        budget: u64,
    },
```

The enum derives `Debug, Clone, PartialEq, Eq` already; the new variants must also be `PartialEq + Eq`. `Vec<String>` and `String` derive `PartialEq` natively, so this works.

- [ ] **Step 3: Add a test**

Append to `crates/heartbit-ghost/src/reply/spam_guard.rs::tests`:

```rust
    #[test]
    fn skip_reason_p1_7_variants_distinct() {
        let a = SkipReason::OwnThreadContinuation;
        let b = SkipReason::BotSuspected { reasons: vec!["test".into()] };
        let c = SkipReason::ConversationDepthExceeded {
            conversation_id: "x".into(),
            count: 2,
            cap: 2,
        };
        let d = SkipReason::DailyBudgetExhausted { used: 100, budget: 50 };
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(c, d);
    }
```

- [ ] **Step 4: Run + format + commit**

```bash
cargo test -p heartbit-ghost --lib reply::spam_guard
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add crates/heartbit-ghost/src/reply/spam_guard.rs
git commit -m "$(cat <<'EOF'
feat(ghost): SkipReason — 4 new P1.7 variants

OwnThreadContinuation, BotSuspected { reasons }, ConversationDepthExceeded
{ conversation_id, count, cap }, DailyBudgetExhausted { used, budget }.
Each captures the audit data needed for the skip log + history record.

heartbit-ghost P1.7a — task 4/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `ThreadDepthGuard` — pure logic + 4 unit tests

**Files:**
- Create: `crates/heartbit-ghost/src/reply/thread_guard.rs`
- Modify: `crates/heartbit-ghost/src/reply/mod.rs` (declare module + re-export)

- [ ] **Step 1: Write the failing tests + skeleton**

Create `crates/heartbit-ghost/src/reply/thread_guard.rs`:

```rust
//! Thread-depth guard — skips a mention when its parent tweet is in
//! our replied set (i.e., this is a continuation of a thread we already
//! engaged with). Catches the dominant AI-to-AI loop shape: another
//! bot replies to our reply, threading on our original tweet.
//!
//! See P1.7 spec §4.

use super::storage::{MentionStore, StoreError};
use super::{Mention, spam_guard::SkipReason};

/// Thread-depth guard — async because it consults `MentionStore`.
pub struct ThreadDepthGuard {
    enabled: bool,
}

impl ThreadDepthGuard {
    /// Construct an enabled guard.
    pub fn new() -> Self {
        Self { enabled: true }
    }

    /// Construct a guard with an explicit enable flag (use `false`
    /// to disable; the guard then always returns `Ok(None)`).
    pub fn with_enabled(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Returns `Some(SkipReason::OwnThreadContinuation)` when the
    /// mention's parent is in our replied set; `None` to proceed.
    /// Errors propagate from the store.
    pub async fn should_skip(
        &self,
        mention: &Mention,
        store: &dyn MentionStore,
    ) -> Result<Option<SkipReason>, StoreError> {
        if !self.enabled {
            return Ok(None);
        }
        let Some(parent_id) = mention.in_reply_to_tweet_id.as_deref() else {
            return Ok(None);
        };
        if store.was_replied(parent_id).await? {
            Ok(Some(SkipReason::OwnThreadContinuation))
        } else {
            Ok(None)
        }
    }
}

impl Default for ThreadDepthGuard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reply::InMemoryMentionStore;
    use chrono::Utc;

    fn fixture_mention(in_reply_to: Option<&str>) -> Mention {
        Mention {
            id: "m1".into(),
            text: "hi".into(),
            author_id: "1".into(),
            author_handle: "x".into(),
            posted_at: Utc::now(),
            in_reply_to_tweet_id: in_reply_to.map(String::from),
            conversation_id: None,
        }
    }

    #[tokio::test]
    async fn skips_when_parent_in_replied_set() {
        let store = InMemoryMentionStore::new();
        store.mark_replied("parent_id").await.unwrap();
        let guard = ThreadDepthGuard::new();
        let m = fixture_mention(Some("parent_id"));
        assert_eq!(
            guard.should_skip(&m, &store).await.unwrap(),
            Some(SkipReason::OwnThreadContinuation)
        );
    }

    #[tokio::test]
    async fn proceeds_when_parent_not_in_replied_set() {
        let store = InMemoryMentionStore::new();
        let guard = ThreadDepthGuard::new();
        let m = fixture_mention(Some("unknown_parent"));
        assert!(guard.should_skip(&m, &store).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn proceeds_when_no_parent_id() {
        let store = InMemoryMentionStore::new();
        store.mark_replied("anything").await.unwrap();
        let guard = ThreadDepthGuard::new();
        let m = fixture_mention(None);
        assert!(guard.should_skip(&m, &store).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn disabled_guard_always_returns_none() {
        let store = InMemoryMentionStore::new();
        store.mark_replied("parent_id").await.unwrap();
        let guard = ThreadDepthGuard::with_enabled(false);
        let m = fixture_mention(Some("parent_id"));
        assert!(guard.should_skip(&m, &store).await.unwrap().is_none());
    }
}
```

- [ ] **Step 2: Wire into `reply/mod.rs`**

In `crates/heartbit-ghost/src/reply/mod.rs`, add (alphabetical placement among existing `pub mod` declarations):

```rust
pub mod thread_guard;
pub use thread_guard::ThreadDepthGuard;
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p heartbit-ghost --lib reply::thread_guard
```

Expected: 4 PASS.

- [ ] **Step 4: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add crates/heartbit-ghost/src/reply/thread_guard.rs crates/heartbit-ghost/src/reply/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): ThreadDepthGuard — skip own-thread continuations

Pure-logic guard backed by MentionStore.was_replied. When a mention's
in_reply_to_tweet_id matches a tweet we already replied to, this is
a continuation of an engaged thread — the dominant AI-to-AI loop
shape. Configurable enable flag. 4 tests cover all paths.

heartbit-ghost P1.7a — task 5/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `BotHeuristicGuard` — 3-signal heuristic + 6 unit tests

**Files:**
- Create: `crates/heartbit-ghost/src/reply/bot_guard.rs`
- Modify: `crates/heartbit-ghost/src/reply/mod.rs` (declare + re-export)

- [ ] **Step 1: Write the file**

Create `crates/heartbit-ghost/src/reply/bot_guard.rs`:

```rust
//! Bot-heuristic guard — skips mentions when ≥ `threshold` of 3
//! signals match: suspicious handle pattern, low follower/following
//! ratio, recent account creation. Conservative (≥2 of 3 by default)
//! to avoid false positives on real humans.
//!
//! See P1.7 spec §5.

use chrono::{DateTime, Duration, Utc};

use super::{Mention, MentionerContext, spam_guard::SkipReason};

/// Configuration for [`BotHeuristicGuard`].
#[derive(Debug, Clone)]
pub struct BotHeuristicConfig {
    /// Substrings that strongly suggest a bot. Case-insensitive
    /// substring match on the author's handle.
    pub suspicious_handle_patterns: Vec<String>,
    /// Minimum follower/following ratio. Skip rule fires when
    /// `followers/following < threshold`. Disabled when
    /// `following_count` is None or 0.
    pub min_follower_following_ratio: f32,
    /// Minimum account age in days. Skip rule fires when the account
    /// is younger.
    pub min_account_age_days: i64,
    /// Number of signals required to trigger a skip (0 disables the
    /// guard entirely, 3 requires all signals).
    pub threshold: usize,
}

impl BotHeuristicConfig {
    /// Sensible defaults.
    pub fn defaults() -> Self {
        Self {
            suspicious_handle_patterns: vec![
                "_bot".into(),
                "_gpt".into(),
                "_ai".into(),
                "ai_".into(),
                "gpt_".into(),
                "bot_".into(),
            ],
            min_follower_following_ratio: 0.05,
            min_account_age_days: 7,
            threshold: 2,
        }
    }
}

impl Default for BotHeuristicConfig {
    fn default() -> Self {
        Self::defaults()
    }
}

/// Bot heuristic guard. Pure logic — no network, no I/O.
pub struct BotHeuristicGuard {
    cfg: BotHeuristicConfig,
}

impl BotHeuristicGuard {
    /// Construct from config.
    pub fn new(cfg: BotHeuristicConfig) -> Self {
        Self { cfg }
    }

    /// Returns `Some(SkipReason::BotSuspected { reasons })` when at
    /// least `threshold` signals match. `None` to proceed.
    pub fn should_skip(
        &self,
        mention: &Mention,
        mentioner: Option<&MentionerContext>,
        now: DateTime<Utc>,
    ) -> Option<SkipReason> {
        if self.cfg.threshold == 0 {
            return None; // disabled
        }
        let mut reasons: Vec<String> = Vec::new();

        // Signal 1: handle suffix/prefix match (always evaluable; uses
        // mention.author_handle even when mentioner is None).
        let handle_lower = mention.author_handle.to_lowercase();
        for pattern in &self.cfg.suspicious_handle_patterns {
            if handle_lower.contains(&pattern.to_lowercase()) {
                reasons.push(format!("handle pattern '{pattern}'"));
                break; // one match per signal
            }
        }

        if let Some(ctx) = mentioner {
            // Signal 2: follower/following ratio.
            if let (Some(followers), Some(following)) = (ctx.follower_count, ctx.following_count) {
                if following > 0 {
                    let ratio = followers as f32 / following as f32;
                    if ratio < self.cfg.min_follower_following_ratio {
                        reasons.push(format!(
                            "follower/following ratio {ratio:.3} < {:.3}",
                            self.cfg.min_follower_following_ratio
                        ));
                    }
                }
            }
            // Signal 3: account age.
            if let Some(created) = ctx.account_created_at {
                let age = now.signed_duration_since(created);
                if age < Duration::days(self.cfg.min_account_age_days) {
                    reasons.push(format!(
                        "account age {} days < {}",
                        age.num_days(),
                        self.cfg.min_account_age_days
                    ));
                }
            }
        }

        if reasons.len() >= self.cfg.threshold {
            Some(SkipReason::BotSuspected { reasons })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reply::TweetSnapshot;

    fn fixture_mention(handle: &str) -> Mention {
        Mention {
            id: "m1".into(),
            text: "hi".into(),
            author_id: "1".into(),
            author_handle: handle.into(),
            posted_at: Utc::now(),
            in_reply_to_tweet_id: None,
            conversation_id: None,
        }
    }

    fn fixture_ctx(
        followers: Option<u64>,
        following: Option<u64>,
        created_at: Option<DateTime<Utc>>,
    ) -> MentionerContext {
        MentionerContext {
            handle: "x".into(),
            bio: None,
            recent_tweets: vec![],
            follower_count: followers,
            following_count: following,
            account_created_at: created_at,
        }
    }

    #[test]
    fn handle_pattern_signal_matches_substring_case_insensitive() {
        let guard = BotHeuristicGuard::new(BotHeuristicConfig {
            threshold: 1, // single signal triggers
            ..BotHeuristicConfig::defaults()
        });
        let m = fixture_mention("ChatGPT_BOT");
        let result = guard.should_skip(&m, None, Utc::now());
        assert!(result.is_some());
        if let Some(SkipReason::BotSuspected { reasons }) = result {
            assert_eq!(reasons.len(), 1);
            assert!(
                reasons[0].contains("_bot") || reasons[0].contains("_gpt"),
                "got: {:?}",
                reasons
            );
        }
    }

    #[test]
    fn follow_ratio_signal_matches_below_threshold() {
        let guard = BotHeuristicGuard::new(BotHeuristicConfig {
            threshold: 1,
            min_follower_following_ratio: 0.05,
            ..BotHeuristicConfig::defaults()
        });
        let m = fixture_mention("normal_user");
        // 10 followers / 1000 following = 0.01 < 0.05
        let ctx = fixture_ctx(Some(10), Some(1000), None);
        let result = guard.should_skip(&m, Some(&ctx), Utc::now());
        assert!(matches!(result, Some(SkipReason::BotSuspected { .. })));
    }

    #[test]
    fn account_age_signal_matches_recent_account() {
        let guard = BotHeuristicGuard::new(BotHeuristicConfig {
            threshold: 1,
            min_account_age_days: 7,
            ..BotHeuristicConfig::defaults()
        });
        let m = fixture_mention("normal_user");
        let now = Utc::now();
        let recent = now - Duration::days(2); // 2-day-old account
        let ctx = fixture_ctx(None, None, Some(recent));
        let result = guard.should_skip(&m, Some(&ctx), now);
        assert!(matches!(result, Some(SkipReason::BotSuspected { .. })));
    }

    #[test]
    fn threshold_2_requires_two_signals() {
        let guard = BotHeuristicGuard::new(BotHeuristicConfig {
            threshold: 2,
            ..BotHeuristicConfig::defaults()
        });
        let m = fixture_mention("normal_user_bot"); // matches "_bot" pattern
        let now = Utc::now();
        // 1-day-old account — 2 signals total.
        let ctx = fixture_ctx(None, None, Some(now - Duration::days(1)));
        let result = guard.should_skip(&m, Some(&ctx), now);
        assert!(matches!(result, Some(SkipReason::BotSuspected { reasons }) if reasons.len() == 2));
    }

    #[test]
    fn single_signal_does_not_skip_at_threshold_2() {
        let guard = BotHeuristicGuard::new(BotHeuristicConfig {
            threshold: 2,
            ..BotHeuristicConfig::defaults()
        });
        let m = fixture_mention("normal_user_bot"); // 1 signal (handle)
        // No follow ratio, no account age — only 1 signal total.
        let result = guard.should_skip(&m, None, Utc::now());
        assert!(result.is_none());
    }

    #[test]
    fn threshold_zero_disables_guard() {
        let guard = BotHeuristicGuard::new(BotHeuristicConfig {
            threshold: 0,
            ..BotHeuristicConfig::defaults()
        });
        let m = fixture_mention("definitely_a_bot_gpt_ai");
        let now = Utc::now();
        let ctx = fixture_ctx(Some(0), Some(10000), Some(now - Duration::days(1)));
        let result = guard.should_skip(&m, Some(&ctx), now);
        assert!(result.is_none(), "threshold=0 should disable; got {result:?}");
    }
}
```

- [ ] **Step 2: Wire into `reply/mod.rs`**

```rust
pub mod bot_guard;
pub use bot_guard::{BotHeuristicConfig, BotHeuristicGuard};
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p heartbit-ghost --lib reply::bot_guard
```

Expected: 6 PASS.

- [ ] **Step 4: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add crates/heartbit-ghost/src/reply/bot_guard.rs crates/heartbit-ghost/src/reply/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): BotHeuristicGuard — 3-signal heuristic with configurable threshold

3 signals: suspicious handle substring (default list: _bot, _gpt, _ai,
ai_, gpt_, bot_), follower/following ratio (default <0.05), account
age (default <7 days). Threshold default 2 of 3 — conservative;
single signal alone doesn't fire. Skipping the guard via threshold=0
is supported. 6 tests cover signal independence, threshold behavior,
case-insensitivity, and the disabled mode.

heartbit-ghost P1.7a — task 6/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `MentionStore` extensions — `replies_in_conversation` + `record_reply_in_conversation`

**Files:**
- Modify: `crates/heartbit-ghost/src/reply/storage.rs` (extend trait + both impls)

- [ ] **Step 1: Read the existing trait + impls**

```bash
sed -n '1,180p' crates/heartbit-ghost/src/reply/storage.rs
```

Note the existing 6 trait methods using `Pin<Box<dyn Future>>`. New methods follow the same pattern.

- [ ] **Step 2: Add the trait methods**

In `crates/heartbit-ghost/src/reply/storage.rs::MentionStore` trait, append two methods:

```rust
    /// Number of replies we've already sent in `conversation_id`.
    /// Used by the conversation-depth guard (P1.7).
    fn replies_in_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<usize, StoreError>> + Send + 'a>>;

    /// Record that we just sent a reply in `conversation_id`.
    /// Called from the daemon's ReplyDraft handler after a successful
    /// `Posted` outcome.
    fn record_reply_in_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>>;
```

- [ ] **Step 3: Add an in-memory counter to `InMemoryInner`**

In the `InMemoryInner` struct (the shared private state for `InMemoryMentionStore`), add a field:

```rust
    /// conversation_id → reply count.
    convo_replies: std::collections::HashMap<String, usize>,
```

(Or whatever the existing struct's field naming convention is — match it.)

- [ ] **Step 4: Implement on `InMemoryMentionStore`**

```rust
    fn replies_in_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<usize, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            Ok(self
                .inner
                .read()
                .await
                .convo_replies
                .get(conversation_id)
                .copied()
                .unwrap_or(0))
        })
    }

    fn record_reply_in_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            *self
                .inner
                .write()
                .await
                .convo_replies
                .entry(conversation_id.to_string())
                .or_insert(0) += 1;
            Ok(())
        })
    }
```

- [ ] **Step 5: Implement on `JsonlMentionStore`**

The JSONL impl appends to a file + replays at open. Add a new event variant:

```rust
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum StoreEvent {
    // ... existing variants ...
    ConversationReply {
        conversation_id: String,
    },
}
```

In `replay()`, add a match arm:
```rust
StoreEvent::ConversationReply { conversation_id } => {
    *g.convo_replies.entry(conversation_id).or_insert(0) += 1;
}
```

In `JsonlMentionStore` impl block, add the new methods (they read the in-memory mirror for `replies_in_conversation`; for `record_reply_in_conversation`, append the event then update the mirror):

```rust
    fn replies_in_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<usize, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            Ok(self
                .inner
                .read()
                .await
                .convo_replies
                .get(conversation_id)
                .copied()
                .unwrap_or(0))
        })
    }

    fn record_reply_in_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            self.append(&StoreEvent::ConversationReply {
                conversation_id: conversation_id.to_string(),
            })
            .await?;
            *self
                .inner
                .write()
                .await
                .convo_replies
                .entry(conversation_id.to_string())
                .or_insert(0) += 1;
            Ok(())
        })
    }
```

`InMemoryInner::convo_replies` is shared between both impls — if not (each has its own private inner struct), add the field to both.

- [ ] **Step 6: Add tests**

Append to `crates/heartbit-ghost/src/reply/storage.rs::tests`:

```rust
    #[tokio::test]
    async fn in_memory_record_then_count_in_conversation_round_trip() {
        let store = InMemoryMentionStore::new();
        assert_eq!(store.replies_in_conversation("c1").await.unwrap(), 0);
        store.record_reply_in_conversation("c1").await.unwrap();
        store.record_reply_in_conversation("c1").await.unwrap();
        store.record_reply_in_conversation("c2").await.unwrap();
        assert_eq!(store.replies_in_conversation("c1").await.unwrap(), 2);
        assert_eq!(store.replies_in_conversation("c2").await.unwrap(), 1);
        assert_eq!(store.replies_in_conversation("nonexistent").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn jsonl_conversation_replies_persist_across_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("conv.jsonl");
        {
            let s1 = JsonlMentionStore::open(&path).await.unwrap();
            s1.record_reply_in_conversation("c1").await.unwrap();
            s1.record_reply_in_conversation("c1").await.unwrap();
        }
        let s2 = JsonlMentionStore::open(&path).await.unwrap();
        assert_eq!(s2.replies_in_conversation("c1").await.unwrap(), 2);
    }
```

- [ ] **Step 7: Run + format + commit**

```bash
cargo test -p heartbit-ghost --lib reply::storage
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add crates/heartbit-ghost/src/reply/storage.rs
git commit -m "$(cat <<'EOF'
feat(ghost): MentionStore — replies_in_conversation + record_reply_in_conversation

Two new methods on the trait for the P1.7 conversation-depth guard.
InMemory uses a HashMap<String, usize>; JSONL adds a ConversationReply
event to the append-only log. Both impls round-trip across reload.
2 new unit tests.

heartbit-ghost P1.7b — task 7/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `ConversationDepthGuard` + 4 unit tests

**Files:**
- Create: `crates/heartbit-ghost/src/reply/conversation_guard.rs`
- Modify: `crates/heartbit-ghost/src/reply/mod.rs`

- [ ] **Step 1: Write the file**

Create `crates/heartbit-ghost/src/reply/conversation_guard.rs`:

```rust
//! Conversation-depth guard — caps reply count per X conversation.
//! Catches third-party-joined threads where each new participant
//! could otherwise drag heartbit into 5+ message exchanges.
//!
//! See P1.7 spec §6.

use super::storage::{MentionStore, StoreError};
use super::{Mention, spam_guard::SkipReason};

/// Conversation-depth guard.
pub struct ConversationDepthGuard {
    cap: usize,
}

impl ConversationDepthGuard {
    /// Construct with `cap`. `cap = 0` disables the guard.
    pub fn new(cap: usize) -> Self {
        Self { cap }
    }

    /// Returns `Some(SkipReason::ConversationDepthExceeded)` when the
    /// conversation already has ≥ `cap` replies from us. `None` when
    /// the cap is 0, the mention has no conversation_id, or the count
    /// is below cap.
    pub async fn should_skip(
        &self,
        mention: &Mention,
        store: &dyn MentionStore,
    ) -> Result<Option<SkipReason>, StoreError> {
        if self.cap == 0 {
            return Ok(None);
        }
        let Some(conversation_id) = mention.conversation_id.as_deref() else {
            return Ok(None);
        };
        let count = store.replies_in_conversation(conversation_id).await?;
        if count >= self.cap {
            Ok(Some(SkipReason::ConversationDepthExceeded {
                conversation_id: conversation_id.to_string(),
                count,
                cap: self.cap,
            }))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reply::InMemoryMentionStore;
    use chrono::Utc;

    fn fixture_mention(conversation_id: Option<&str>) -> Mention {
        Mention {
            id: "m1".into(),
            text: "hi".into(),
            author_id: "1".into(),
            author_handle: "x".into(),
            posted_at: Utc::now(),
            in_reply_to_tweet_id: None,
            conversation_id: conversation_id.map(String::from),
        }
    }

    #[tokio::test]
    async fn skips_at_cap() {
        let store = InMemoryMentionStore::new();
        store.record_reply_in_conversation("c1").await.unwrap();
        store.record_reply_in_conversation("c1").await.unwrap();
        let guard = ConversationDepthGuard::new(2);
        let m = fixture_mention(Some("c1"));
        let result = guard.should_skip(&m, &store).await.unwrap();
        match result {
            Some(SkipReason::ConversationDepthExceeded { conversation_id, count, cap }) => {
                assert_eq!(conversation_id, "c1");
                assert_eq!(count, 2);
                assert_eq!(cap, 2);
            }
            other => panic!("expected ConversationDepthExceeded, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn proceeds_below_cap() {
        let store = InMemoryMentionStore::new();
        store.record_reply_in_conversation("c1").await.unwrap();
        let guard = ConversationDepthGuard::new(2);
        let m = fixture_mention(Some("c1"));
        assert!(guard.should_skip(&m, &store).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn proceeds_when_conversation_id_absent() {
        let store = InMemoryMentionStore::new();
        let guard = ConversationDepthGuard::new(2);
        let m = fixture_mention(None);
        assert!(guard.should_skip(&m, &store).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn cap_zero_disables() {
        let store = InMemoryMentionStore::new();
        for _ in 0..10 {
            store.record_reply_in_conversation("c1").await.unwrap();
        }
        let guard = ConversationDepthGuard::new(0);
        let m = fixture_mention(Some("c1"));
        assert!(guard.should_skip(&m, &store).await.unwrap().is_none());
    }
}
```

- [ ] **Step 2: Wire into `reply/mod.rs`**

```rust
pub mod conversation_guard;
pub use conversation_guard::ConversationDepthGuard;
```

- [ ] **Step 3: Run tests + format + commit**

```bash
cargo test -p heartbit-ghost --lib reply::conversation_guard
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add crates/heartbit-ghost/src/reply/conversation_guard.rs crates/heartbit-ghost/src/reply/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): ConversationDepthGuard — cap replies per X conversation

Async guard backed by MentionStore.replies_in_conversation.
cap=0 disables; cap=N skips when reply count ≥ N. Skip reason
captures conversation_id, count, cap for the audit log. 4 tests
cover at-cap, below-cap, no-conversation-id, and disabled.

heartbit-ghost P1.7b — task 8/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `DailyTokenBudget` trait + `InMemoryDailyBudget` impl

**Files:**
- Create: `crates/heartbit-ghost/src/reply/budget.rs`
- Modify: `crates/heartbit-ghost/src/reply/mod.rs`

- [ ] **Step 1: Write the trait + InMemory impl**

Create `crates/heartbit-ghost/src/reply/budget.rs`:

```rust
//! Daily token budget tracker for the mentions pipeline.
//!
//! Tracks total LLM tokens spent per persona per UTC day. The
//! [`DailyBudgetGuard`] (Task 11) consults this to short-circuit
//! mention drafts when the daily cap is hit.
//!
//! See P1.7 spec §7.

use std::future::Future;
use std::pin::Pin;

use chrono::{NaiveDate, Utc};

/// Errors raised by [`DailyTokenBudget`] implementations.
#[derive(Debug, thiserror::Error)]
pub enum BudgetError {
    /// I/O failure (file not readable, write failed, etc.).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSONL parse failure on replay.
    #[error("parse: {0}")]
    Parse(String),
}

/// Persistent storage for daily-budget accounting.
pub trait DailyTokenBudget: Send + Sync {
    /// Total tokens recorded for `persona` on the current UTC day.
    fn usage_today<'a>(
        &'a self,
        persona: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<u64, BudgetError>> + Send + 'a>>;

    /// Append `tokens` to today's accumulator for `persona`.
    fn record_usage<'a>(
        &'a self,
        persona: &'a str,
        tokens: u64,
    ) -> Pin<Box<dyn Future<Output = Result<(), BudgetError>> + Send + 'a>>;
}

/// Volatile in-memory budget tracker. For tests and dev runs.
pub struct InMemoryDailyBudget {
    inner: tokio::sync::RwLock<InMemoryInner>,
}

#[derive(Default)]
struct InMemoryInner {
    /// (date, persona) → tokens used.
    usage: std::collections::HashMap<(NaiveDate, String), u64>,
}

impl Default for InMemoryDailyBudget {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryDailyBudget {
    /// Construct an empty budget tracker.
    pub fn new() -> Self {
        Self {
            inner: tokio::sync::RwLock::new(InMemoryInner::default()),
        }
    }
}

impl DailyTokenBudget for InMemoryDailyBudget {
    fn usage_today<'a>(
        &'a self,
        persona: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<u64, BudgetError>> + Send + 'a>> {
        Box::pin(async move {
            let today = Utc::now().date_naive();
            Ok(self
                .inner
                .read()
                .await
                .usage
                .get(&(today, persona.to_string()))
                .copied()
                .unwrap_or(0))
        })
    }

    fn record_usage<'a>(
        &'a self,
        persona: &'a str,
        tokens: u64,
    ) -> Pin<Box<dyn Future<Output = Result<(), BudgetError>> + Send + 'a>> {
        Box::pin(async move {
            let today = Utc::now().date_naive();
            *self
                .inner
                .write()
                .await
                .usage
                .entry((today, persona.to_string()))
                .or_insert(0) += tokens;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_record_then_read_round_trip() {
        let b = InMemoryDailyBudget::new();
        assert_eq!(b.usage_today("p").await.unwrap(), 0);
        b.record_usage("p", 100).await.unwrap();
        b.record_usage("p", 200).await.unwrap();
        assert_eq!(b.usage_today("p").await.unwrap(), 300);
    }

    #[tokio::test]
    async fn in_memory_per_persona_isolation() {
        let b = InMemoryDailyBudget::new();
        b.record_usage("a", 50).await.unwrap();
        b.record_usage("b", 100).await.unwrap();
        assert_eq!(b.usage_today("a").await.unwrap(), 50);
        assert_eq!(b.usage_today("b").await.unwrap(), 100);
        assert_eq!(b.usage_today("nonexistent").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn in_memory_zero_tokens_recorded_is_idempotent() {
        let b = InMemoryDailyBudget::new();
        b.record_usage("p", 0).await.unwrap();
        b.record_usage("p", 0).await.unwrap();
        assert_eq!(b.usage_today("p").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn in_memory_usage_today_isolates_by_date_in_inner() {
        // We can't easily mock Utc::now(), but we can verify that
        // entries keyed by yesterday's date don't show up in today's
        // count.
        let b = InMemoryDailyBudget::new();
        let yesterday = Utc::now().date_naive() - chrono::Duration::days(1);
        b.inner
            .write()
            .await
            .usage
            .insert((yesterday, "p".to_string()), 999);
        assert_eq!(b.usage_today("p").await.unwrap(), 0, "yesterday must not leak");
    }
}
```

- [ ] **Step 2: Wire into `reply/mod.rs`**

```rust
pub mod budget;
pub use budget::{BudgetError, DailyTokenBudget, InMemoryDailyBudget};
```

- [ ] **Step 3: Run + format + commit**

```bash
cargo test -p heartbit-ghost --lib reply::budget
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add crates/heartbit-ghost/src/reply/budget.rs crates/heartbit-ghost/src/reply/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): DailyTokenBudget trait + InMemoryDailyBudget impl

Trait: usage_today(persona) -> u64, record_usage(persona, tokens).
Implicit reset at UTC midnight (date-keyed map). InMemory uses
RwLock<HashMap<(NaiveDate, String), u64>>. 4 tests cover round-trip,
per-persona isolation, zero-token idempotence, date isolation.

heartbit-ghost P1.7c — task 9/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `JsonlDailyBudget` impl

**Files:**
- Modify: `crates/heartbit-ghost/src/reply/budget.rs` (add JSONL impl + tests)

- [ ] **Step 1: Append the JSONL impl**

In `crates/heartbit-ghost/src/reply/budget.rs`, after the `InMemoryDailyBudget` impl, append:

```rust
use std::path::PathBuf;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct UsageLine {
    date: NaiveDate,
    persona: String,
    tokens: u64,
}

/// JSONL append-only daily-budget tracker. Replays the file at
/// [`open`] time into an in-memory mirror; subsequent writes both
/// append to the file and update the mirror.
pub struct JsonlDailyBudget {
    path: PathBuf,
    inner: tokio::sync::RwLock<InMemoryInner>,
}

impl JsonlDailyBudget {
    /// Open or create the JSONL store at `path`.
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self, BudgetError> {
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

    async fn replay(&self) -> Result<(), BudgetError> {
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
            let parsed: UsageLine = serde_json::from_str(line)
                .map_err(|e| BudgetError::Parse(format!("line {line:?}: {e}")))?;
            *g.usage
                .entry((parsed.date, parsed.persona))
                .or_insert(0) += parsed.tokens;
        }
        Ok(())
    }

    async fn append(&self, line: &UsageLine) -> Result<(), BudgetError> {
        use tokio::io::AsyncWriteExt;
        let serialized =
            serde_json::to_string(line).map_err(|e| BudgetError::Parse(format!("{e}")))?;
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

impl DailyTokenBudget for JsonlDailyBudget {
    fn usage_today<'a>(
        &'a self,
        persona: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<u64, BudgetError>> + Send + 'a>> {
        Box::pin(async move {
            let today = Utc::now().date_naive();
            Ok(self
                .inner
                .read()
                .await
                .usage
                .get(&(today, persona.to_string()))
                .copied()
                .unwrap_or(0))
        })
    }

    fn record_usage<'a>(
        &'a self,
        persona: &'a str,
        tokens: u64,
    ) -> Pin<Box<dyn Future<Output = Result<(), BudgetError>> + Send + 'a>> {
        Box::pin(async move {
            let today = Utc::now().date_naive();
            let line = UsageLine {
                date: today,
                persona: persona.to_string(),
                tokens,
            };
            self.append(&line).await?;
            *self
                .inner
                .write()
                .await
                .usage
                .entry((today, persona.to_string()))
                .or_insert(0) += tokens;
            Ok(())
        })
    }
}
```

- [ ] **Step 2: Re-export from `reply/mod.rs`**

```rust
pub use budget::JsonlDailyBudget;
```

(Or add to the existing `pub use budget::{...}` block.)

- [ ] **Step 3: Add tests**

Append to `crates/heartbit-ghost/src/reply/budget.rs::tests`:

```rust
    #[tokio::test]
    async fn jsonl_record_then_read_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("budget.jsonl");
        let b = JsonlDailyBudget::open(&path).await.unwrap();
        b.record_usage("p", 100).await.unwrap();
        b.record_usage("p", 200).await.unwrap();
        assert_eq!(b.usage_today("p").await.unwrap(), 300);
    }

    #[tokio::test]
    async fn jsonl_reload_preserves_today_and_yesterday() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("budget.jsonl");
        {
            let b = JsonlDailyBudget::open(&path).await.unwrap();
            b.record_usage("p", 100).await.unwrap();
        }
        // Reload — usage_today should still see today's record.
        let b2 = JsonlDailyBudget::open(&path).await.unwrap();
        assert_eq!(b2.usage_today("p").await.unwrap(), 100);
    }

    #[tokio::test]
    async fn jsonl_handles_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.jsonl");
        let b = JsonlDailyBudget::open(&path).await.unwrap();
        assert_eq!(b.usage_today("p").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn jsonl_per_persona_isolation_across_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("budget.jsonl");
        {
            let b = JsonlDailyBudget::open(&path).await.unwrap();
            b.record_usage("a", 50).await.unwrap();
            b.record_usage("b", 100).await.unwrap();
        }
        let b2 = JsonlDailyBudget::open(&path).await.unwrap();
        assert_eq!(b2.usage_today("a").await.unwrap(), 50);
        assert_eq!(b2.usage_today("b").await.unwrap(), 100);
    }
```

- [ ] **Step 4: Run + format + commit**

```bash
cargo test -p heartbit-ghost --lib reply::budget
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add crates/heartbit-ghost/src/reply/budget.rs crates/heartbit-ghost/src/reply/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): JsonlDailyBudget — append-only on-disk impl

JSONL append + in-memory replay at open. Each line is
{date, persona, tokens}; date is stringly NaiveDate (YYYY-MM-DD).
Mirrors JsonlMentionStore's shape exactly. 4 new tests cover
round-trip, reload, missing-file, per-persona isolation.

heartbit-ghost P1.7c — task 10/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `DailyBudgetGuard` — thin wrapper

**Files:**
- Create: `crates/heartbit-ghost/src/reply/budget_guard.rs`
- Modify: `crates/heartbit-ghost/src/reply/mod.rs`

- [ ] **Step 1: Write the guard**

Create `crates/heartbit-ghost/src/reply/budget_guard.rs`:

```rust
//! Daily-budget guard — short-circuits mention drafting when the
//! persona's daily LLM token usage has reached the configured cap.
//! Default budget is `None` (unlimited); operator must opt in.
//!
//! See P1.7 spec §7.

use super::budget::{BudgetError, DailyTokenBudget};
use super::spam_guard::SkipReason;

/// Daily-budget guard.
pub struct DailyBudgetGuard {
    /// `None` means unlimited (always returns `Ok(None)`).
    budget: Option<u64>,
}

impl DailyBudgetGuard {
    /// Construct from a configured budget. `None` disables the guard.
    pub fn new(budget: Option<u64>) -> Self {
        Self { budget }
    }

    /// Returns `Some(SkipReason::DailyBudgetExhausted)` when the
    /// persona's usage today is at or above the configured budget.
    pub async fn should_skip(
        &self,
        persona: &str,
        tracker: &dyn DailyTokenBudget,
    ) -> Result<Option<SkipReason>, BudgetError> {
        let Some(budget) = self.budget else {
            return Ok(None);
        };
        let used = tracker.usage_today(persona).await?;
        if used >= budget {
            Ok(Some(SkipReason::DailyBudgetExhausted { used, budget }))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reply::InMemoryDailyBudget;

    #[tokio::test]
    async fn proceeds_when_below_budget() {
        let tracker = InMemoryDailyBudget::new();
        tracker.record_usage("p", 100).await.unwrap();
        let guard = DailyBudgetGuard::new(Some(500));
        assert!(guard.should_skip("p", &tracker).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn skips_when_at_or_above_budget() {
        let tracker = InMemoryDailyBudget::new();
        tracker.record_usage("p", 500).await.unwrap();
        let guard = DailyBudgetGuard::new(Some(500));
        let result = guard.should_skip("p", &tracker).await.unwrap();
        match result {
            Some(SkipReason::DailyBudgetExhausted { used, budget }) => {
                assert_eq!(used, 500);
                assert_eq!(budget, 500);
            }
            other => panic!("expected DailyBudgetExhausted, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn none_budget_always_proceeds() {
        let tracker = InMemoryDailyBudget::new();
        for _ in 0..1000 {
            tracker.record_usage("p", 100_000).await.unwrap();
        }
        let guard = DailyBudgetGuard::new(None);
        assert!(guard.should_skip("p", &tracker).await.unwrap().is_none());
    }
}
```

- [ ] **Step 2: Wire into `reply/mod.rs`**

```rust
pub mod budget_guard;
pub use budget_guard::DailyBudgetGuard;
```

- [ ] **Step 3: Run + format + commit**

```bash
cargo test -p heartbit-ghost --lib reply::budget_guard
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add crates/heartbit-ghost/src/reply/budget_guard.rs crates/heartbit-ghost/src/reply/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): DailyBudgetGuard — thin wrapper around DailyTokenBudget

Skips when usage_today(persona) >= budget. None (the default) means
unlimited — guard always returns None. 3 tests cover under-budget,
at-budget (exact equality boundary), and the unlimited path.

heartbit-ghost P1.7c — task 11/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Wire 4 guards into `handle_mention_poll` + extend config + CLI wiring

**Files:**
- Modify: `crates/heartbit-ghost/src/tools/user.rs` (or wherever `MentionerContext` is fetched, to populate `following_count` and `account_created_at`)
- Modify: `crates/heartbit-core/src/config/daemon.rs` (extend `PersonaMentionsConfig` with new fields)
- Modify: `crates/heartbit/src/daemon/mention_context.rs` (extend `PersonaMentionEntry`)
- Modify: `crates/heartbit/src/daemon/mention_poll_handler.rs` (chain the 4 guards)
- Modify: `crates/heartbit/src/daemon/reply_draft_handler.rs` (record_reply_in_conversation + record_usage on Posted)
- Modify: `crates/heartbit-cli/src/daemon/mod.rs` (CLI wiring)

This is the integration task. Several sub-steps.

### 12a. Extend `PersonaMentionsConfig`

In `crates/heartbit-core/src/config/daemon.rs::PersonaMentionsConfig`:

```rust
    /// P1.7: enable the thread-depth guard (default true).
    #[serde(default = "super::default_true")]
    pub enable_thread_depth_guard: bool,
    /// P1.7: enable the bot-heuristic guard (default true).
    #[serde(default = "super::default_true")]
    pub enable_bot_heuristic_guard: bool,
    /// P1.7: bot-heuristic substring patterns. Empty list uses the
    /// default list from `BotHeuristicConfig::defaults`.
    #[serde(default)]
    pub suspicious_handle_patterns: Vec<String>,
    /// P1.7: minimum follower/following ratio (default 0.05).
    #[serde(default = "default_min_follower_following_ratio")]
    pub min_follower_following_ratio: f32,
    /// P1.7: minimum account age in days (default 7).
    #[serde(default = "default_min_account_age_days")]
    pub min_account_age_days: i64,
    /// P1.7: number of bot-heuristic signals required to skip
    /// (default 2). Set to 0 to disable, 3 to require all.
    #[serde(default = "default_bot_heuristic_threshold")]
    pub bot_heuristic_threshold: usize,
    /// P1.7: max replies per X conversation (default 2). Set to 0
    /// to disable.
    #[serde(default = "default_per_conversation_max_replies")]
    pub per_conversation_max_replies: usize,
    /// P1.7: per-persona daily LLM token cap. None disables (unlimited).
    #[serde(default)]
    pub daily_token_budget: Option<u64>,
    /// P1.7: budget store backend ("in_memory" | "jsonl"). Default
    /// "in_memory".
    #[serde(default = "default_budget_store")]
    pub budget_store: String,
    /// P1.7: jsonl path (only used when budget_store == "jsonl").
    #[serde(default)]
    pub budget_path: Option<String>,
```

Add the helper functions at the bottom of the file:

```rust
fn default_min_follower_following_ratio() -> f32 {
    0.05
}
fn default_min_account_age_days() -> i64 {
    7
}
fn default_bot_heuristic_threshold() -> usize {
    2
}
fn default_per_conversation_max_replies() -> usize {
    2
}
fn default_budget_store() -> String {
    "in_memory".into()
}
```

`super::default_true` already exists in the parent module.

Add a deserialize test to confirm defaults:

```rust
    #[test]
    fn persona_mentions_config_p1_7_defaults() {
        let toml = r#"
[[posts]]
persona = "heartbit-ghost:x"
user_id = "1234567890"
"#;
        // Adapt this shim to whatever the existing test pattern uses.
        // The key assertions:
        //   enable_thread_depth_guard == true
        //   enable_bot_heuristic_guard == true
        //   bot_heuristic_threshold == 2
        //   per_conversation_max_replies == 2
        //   daily_token_budget is None
        //   budget_store == "in_memory"
    }
```

(Adapt the shim to the existing test's pattern in this file. The point is: a minimal config has the new fields at their defaults.)

### 12b. Extend `PersonaMentionEntry`

In `crates/heartbit/src/daemon/mention_context.rs::PersonaMentionEntry`:

```rust
pub struct PersonaMentionEntry {
    // ... existing fields ...

    /// P1.7: thread-depth guard enable flag.
    pub enable_thread_depth_guard: bool,
    /// P1.7: bot-heuristic guard config (None disables).
    pub bot_heuristic: Option<heartbit_ghost::reply::BotHeuristicConfig>,
    /// P1.7: per-conversation reply cap (0 disables).
    pub per_conversation_max_replies: usize,
    /// P1.7: shared daily-budget tracker.
    pub budget_tracker: std::sync::Arc<dyn heartbit_ghost::reply::DailyTokenBudget>,
    /// P1.7: persona's daily token budget (None = unlimited).
    pub daily_token_budget: Option<u64>,
}
```

### 12c. Wire guards into `handle_mention_poll`

In `crates/heartbit/src/daemon/mention_poll_handler.rs::handle_mention_poll`, after the existing `SpamGuard::should_skip` check and BEFORE dispatching `DaemonCommand::ReplyDraft`, chain the 4 new guards. Construct each guard from the entry's config (passed in via `MentionPollDeps`).

Pseudo:
```rust
// Existing:
if let Some(reason) = spam_guard.should_skip(...) {
    store.mark_replied(&mention.id).await?;
    tracing::info!(reason = ?reason, mention_id = %mention.id, "mention skipped (P1.5 spam_guard)");
    continue;
}

// P1.7 thread depth:
let thread_guard = ThreadDepthGuard::with_enabled(deps.enable_thread_depth_guard);
if let Some(reason) = thread_guard.should_skip(&mention, store).await? {
    store.mark_replied(&mention.id).await?;
    tracing::info!(reason = ?reason, mention_id = %mention.id, "mention skipped (P1.7 thread_depth)");
    continue;
}

// P1.7 bot heuristic:
if let Some(cfg) = &deps.bot_heuristic {
    let bot_guard = BotHeuristicGuard::new(cfg.clone());
    if let Some(reason) = bot_guard.should_skip(&mention, mentioner_context.as_ref(), Utc::now()) {
        store.mark_replied(&mention.id).await?;
        tracing::info!(reason = ?reason, mention_id = %mention.id, "mention skipped (P1.7 bot_heuristic)");
        continue;
    }
}

// P1.7 conversation depth:
let convo_guard = ConversationDepthGuard::new(deps.per_conversation_max_replies);
if let Some(reason) = convo_guard.should_skip(&mention, store).await? {
    store.mark_replied(&mention.id).await?;
    tracing::info!(reason = ?reason, mention_id = %mention.id, "mention skipped (P1.7 conversation_depth)");
    continue;
}

// P1.7 daily budget:
let budget_guard = DailyBudgetGuard::new(deps.daily_token_budget);
if let Some(reason) = budget_guard.should_skip(persona, deps.budget_tracker.as_ref()).await? {
    store.mark_replied(&mention.id).await?;
    tracing::warn!(reason = ?reason, persona, "mention skipped (P1.7 daily_budget) — replies suspended until UTC midnight");
    continue;
}
```

The `mentioner_context` is fetched separately by the existing P1.5 mention-poll code (or it's None when not yet fetched). If the bot-heuristic guard's signals 2/3 require `MentionerContext` and the handler doesn't fetch it pre-guard, the guard handles `None` gracefully (only signal 1 fires from `mention.author_handle` alone). Spec §5 confirms this.

### 12d. Record on `Posted` outcome

In `crates/heartbit/src/daemon/reply_draft_handler.rs::handle_reply_draft`, AFTER the reply pipeline returns, on `ReplyOutcome::Posted` outcome, increment the conversation counter AND record token usage:

```rust
// After: let output = run_reply_pipeline(cfg).await?;

// P1.7: record per-conversation reply count on success.
if matches!(output.outcome, ReplyOutcome::Posted { .. }) {
    if let Some(conversation_id) = mention.conversation_id.as_deref() {
        if let Err(e) = deps.store.record_reply_in_conversation(conversation_id).await {
            tracing::warn!(error = %e, "record_reply_in_conversation failed");
        }
    }
}

// P1.7: record token usage for the daily budget tracker (regardless
// of outcome — even Skipped/TimedOut runs paid the LLM cost).
let total_tokens = output.usage_summary.input_tokens as u64
    + output.usage_summary.output_tokens as u64
    + output.usage_summary.reasoning_tokens as u64;
if total_tokens > 0 {
    if let Err(e) = deps.budget_tracker.record_usage(persona_name, total_tokens).await {
        tracing::warn!(error = %e, "budget record_usage failed");
    }
}
```

`deps` is `ReplyDraftDeps`. Add `budget_tracker: Arc<dyn DailyTokenBudget>` to the struct (and propagate through the dispatcher arm in `daemon/core.rs`'s `ReplyDraft` arm).

### 12e. CLI wiring

In `crates/heartbit-cli/src/daemon/mod.rs::build_mention_context`, when constructing `PersonaMentionEntry`, populate the new fields from `cfg`. Also instantiate the budget tracker:

```rust
let budget_tracker: std::sync::Arc<dyn heartbit_ghost::reply::DailyTokenBudget> =
    match cfg.budget_store.as_str() {
        "in_memory" => std::sync::Arc::new(heartbit_ghost::reply::InMemoryDailyBudget::new()),
        "jsonl" => {
            let path = cfg.budget_path.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "persona_mentions persona='{}' uses budget_store='jsonl' but no budget_path",
                    cfg.persona
                )
            })?;
            let path = expand_tilde(path)?;
            std::sync::Arc::new(
                heartbit_ghost::reply::JsonlDailyBudget::open(&path)
                    .await
                    .with_context(|| format!("open budget jsonl at {}", path.display()))?,
            )
        }
        other => anyhow::bail!("unknown budget_store '{}'", other),
    };

let bot_heuristic = if cfg.enable_bot_heuristic_guard && cfg.bot_heuristic_threshold > 0 {
    Some(heartbit_ghost::reply::BotHeuristicConfig {
        suspicious_handle_patterns: if cfg.suspicious_handle_patterns.is_empty() {
            heartbit_ghost::reply::BotHeuristicConfig::defaults().suspicious_handle_patterns
        } else {
            cfg.suspicious_handle_patterns.clone()
        },
        min_follower_following_ratio: cfg.min_follower_following_ratio,
        min_account_age_days: cfg.min_account_age_days,
        threshold: cfg.bot_heuristic_threshold,
    })
} else {
    None
};
```

Then build the entry with these fields.

### 12f. Integration test

In `crates/heartbit/src/daemon/mention_poll_handler.rs::tests`, add ONE composite test:

```rust
#[tokio::test]
async fn mention_poll_with_4_guards_filters_mixed_fixture() {
    // 6-mention fixture covering all skip paths plus a happy mention.
    // Verifies exactly 1 ReplyDraft is dispatched.
    //
    // Mentions:
    // 1. Happy path — passes all guards
    // 2. Self-reply — caught by P1.5 SpamGuard
    // 3. Own-thread continuation — caught by ThreadDepthGuard
    // 4. Bot suspect — caught by BotHeuristicGuard (handle "_bot" + low ratio)
    // 5. Conversation-cap exceeded — caught by ConversationDepthGuard
    // 6. Budget-exhausted state — caught by DailyBudgetGuard
    //
    // Each guard should fire on exactly one mention; the producer
    // queue should contain exactly 1 ReplyDraft for the happy mention.
}
```

Implementer expands the body using the existing `mention_poll_handler` test patterns. Reference: `crates/heartbit/src/daemon/mention_poll_handler.rs::tests` for fixture setup, mock store, mock twitter tool, etc.

- [ ] **Step 1: Apply 12a (PersonaMentionsConfig fields)**
- [ ] **Step 2: Apply 12b (PersonaMentionEntry fields)**
- [ ] **Step 3: Apply 12c (handle_mention_poll guard chain)**
- [ ] **Step 4: Apply 12d (handle_reply_draft accounting)**
- [ ] **Step 5: Apply 12e (CLI wiring)**
- [ ] **Step 6: Add the composite integration test**
- [ ] **Step 7: Build + clippy + format + test**

```bash
cargo build --workspace --features daemon
cargo test --workspace --features daemon
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(daemon+cli): wire 4 P1.7 guards into mention pipeline

- PersonaMentionsConfig gains 9 fields for guard configuration
  (enable flags, bot heuristic thresholds, per-conversation cap,
  daily budget cap, budget store backend).
- PersonaMentionEntry holds per-persona guard state (configs +
  shared budget_tracker Arc).
- handle_mention_poll chains: existing P1.5 SpamGuard →
  ThreadDepthGuard → BotHeuristicGuard → ConversationDepthGuard →
  DailyBudgetGuard. Each guard's skip is logged at info or warn
  (budget exhaustion is warn).
- handle_reply_draft increments record_reply_in_conversation on
  Posted outcomes and always records token usage to the budget
  tracker (Skipped/TimedOut etc. still paid the cost).
- CLI build_mention_context constructs InMemoryDailyBudget or
  JsonlDailyBudget per config; resolves budget_path with tilde
  expansion.
- One composite integration test covers the 4-guard filter path
  with a 6-mention fixture (happy + 5 skip variants).

heartbit-ghost P1.7d — task 12/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Acceptance — quality gate + manual setup + live test

**Files:** none modified.

- [ ] **Step 1: Full quality gate**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --features daemon 2>&1 | tail -25
```

Expected: all clean. Tests count: previous baseline (~4143) + ~28 new tests across `reply::*` + `daemon::*`.

- [ ] **Step 2: Build the release binary**

```bash
cargo build --release --bin heartbit 2>&1 | tail -3
```

Expected: `Finished release …`.

- [ ] **Step 3: Operator-side setup (manual; not part of plan deliverable)**

Update `daemon-dev.toml` with the new P1.7 fields under each persona's mention block. Conservative test config:

```toml
[[daemon.persona_mentions]]
persona = "heartbit-ghost:x"
enabled = true
poll_interval_seconds = 300
user_id = "1234567890"
candidates_per_reply = 2
mention_store = "jsonl"
mention_store_path = "~/.heartbit/ghost/mentions/heartbit-ghost-x.jsonl"

# P1.7 fields:
enable_thread_depth_guard = true
enable_bot_heuristic_guard = true
bot_heuristic_threshold = 2
per_conversation_max_replies = 2
daily_token_budget = 50000          # 50k tokens/day for testing
budget_store = "jsonl"
budget_path = "~/.heartbit/ghost/budgets/heartbit-ghost-x.jsonl"
```

- [ ] **Step 4: Live test — guards fire correctly**

1. Restart the daemon: `./target/release/heartbit daemon --config daemon-dev.toml`
2. Wait for the next cron tick.
3. Verify in logs: each mention gets one `info`/`warn` log per skip with the reason.
4. **Self-reply test**: have heartbit reply to a mention (Telegram pick). Then have the same author reply to heartbit's reply. Wait for the next mention poll. Verify `info!` log "mention skipped (P1.7 thread_depth)".
5. **Bot heuristic test**: ask a colleague with a `_bot`-suffixed handle to mention you. Verify `info!` log "mention skipped (P1.7 bot_heuristic)" (if their account also matches a second signal).
6. **Conversation depth test**: have a conversation with 2 replies. The 3rd mention from anyone in that conversation triggers `info!` log "mention skipped (P1.7 conversation_depth)".
7. **Budget exhaustion test**: set `daily_token_budget = 1000` (very low). Reply to 2-3 mentions. The next mention triggers `warn!` log "mention skipped (P1.7 daily_budget) — replies suspended until UTC midnight".

- [ ] **Step 5: Verify the JSONL stores**

```bash
cat ~/.heartbit/ghost/mentions/heartbit-ghost-x.jsonl | tail -5
cat ~/.heartbit/ghost/budgets/heartbit-ghost-x.jsonl | tail -5
```

Mentions store should have `ConversationReply` entries; budget store should have `{date, persona, tokens}` entries summing to today's usage.

- [ ] **Step 6: Final merge / PR**

Once Steps 4-5 pass, finish via the **superpowers:finishing-a-development-branch** skill.

---

## Self-review — pre-execution

- **Spec coverage:** every section maps to at least one task. §3 4-guard architecture (Tasks 5, 6, 8, 11), §4 ThreadDepthGuard (Task 5), §5 BotHeuristicGuard (Task 6), §6 ConversationDepthGuard (Tasks 7+8), §7 DailyTokenBudget + DailyBudgetGuard (Tasks 9, 10, 11), §8 SkipReason taxonomy (Task 4), §9 Files (Tasks 1-12), §10 Test plan (each task ships its own + acceptance), §11 Hors scope (no tasks), §12 Risks (mitigations applied), §13 sub-phases (1-5 / 6-8 / 9-11 / 12-13 ≈ a/b/c/d), §14 open questions (none).

- **Placeholder scan:** Task 12 has six numbered sub-steps (12a-12f) that lay out the integration. No "TBD"/"TODO" — each sub-step has a code snippet or a clear pointer to the existing reference pattern. The composite integration test in 12f points at `mention_poll_handler::tests` for the verbatim fixture pattern, mirroring the P1.5 task 5/10 plan style.

- **Type consistency:**
  - `SkipReason` variants (Task 4) match the names referenced in Tasks 5, 6, 8, 11.
  - `MentionStore` method signatures (Task 7) match the calls in Tasks 5 and 8.
  - `DailyTokenBudget` trait signature (Task 9) matches `DailyBudgetGuard` (Task 11) and `handle_reply_draft` recording (Task 12d).
  - `Mention.conversation_id` (Task 1) is consumed in Task 8 and accumulated in Task 7.
  - `MentionerContext` extensions (Task 2) are consumed by `BotHeuristicGuard` (Task 6).
  - `BotHeuristicConfig` field names (Task 6) match what the CLI builds in Task 12e.

- **Cross-crate dependency direction:** All new code stays within heartbit-ghost (guards) and heartbit (umbrella daemon). heartbit-core gets new `PersonaMentionsConfig` fields (additive) only.

- **Branch state:** the spec is already committed on `feat/heartbit-ghost-p1.7-loop-protection` (commit `06e604d`). The branch was rebased onto the post-P1.5+P1.6 main; Tasks 1-13 land on top.

- **Known open questions for the implementer:**
  - **Where is `MentionerContext` populated**: P1.5 task 10 mentions `fetch_mention_context` in the handler; that helper needs to fetch `following_count` (already in `public_metrics`) and `account_created_at` (`user.fields=created_at`). Verify by reading `crates/heartbit/src/daemon/mention_poll_handler.rs` — the fetch call in P1.5 may already include these via `user.fields=public_metrics` but probably doesn't yet. Add `created_at` and ensure `following_count` is plumbed through.
  - **Token usage source**: the spec says `output.usage_summary.input_tokens + output_tokens + reasoning_tokens`. This excludes cache tokens. Sufficient for V1; if cache tokens are consequential, sum all 5 `TokenUsage` fields.
