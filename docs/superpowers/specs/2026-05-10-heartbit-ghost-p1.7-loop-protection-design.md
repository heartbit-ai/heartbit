# heartbit-ghost P1.7 — Loop & cost protection design

## 1. Goal

Prevent AI-to-AI mention loops and cap LLM cost for the proactive replies pipeline. Adds four orthogonal guards to the existing P1.5 mentions pipeline. Each guard short-circuits cheaply (no LLM call) when its rule fires, and each is independently configurable per `[[daemon.persona_mentions]]` block.

The guards complement (not replace) the existing P1.5 `SpamGuard` and the calibration-mode Telegram review gate. The Telegram gate remains the operator's primary defense; the new guards add automatic protection that runs before a draft is even produced.

## 2. Dependency

P1.7 builds on the **P1.5 reactive replies pipeline** (`MentionStore`, `SpamGuard`, `handle_mention_poll`, `Mention` struct). P1.5's PR (#10) must merge to main before P1.7 implementation begins. The spec can be written independently, but the implementation plan and tasks reference P1.5 types directly.

## 3. The four guards

Evaluation order (after the existing P1.5 `SpamGuard` runs):

```
SpamGuard (P1.5: self-reply, stale parent, low-effort, per-author rate, too-short)
   ↓
ThreadDepthGuard (P1.7-A) — skip if mention's parent is already in our replied set
   ↓
BotHeuristicGuard (P1.7-B) — skip if ≥2 of: handle suffix, follow ratio, account age
   ↓
ConversationDepthGuard (P1.7-C) — skip if conversation_id reply count ≥ cap
   ↓
DailyBudgetGuard (P1.7-D) — skip if persona's daily token usage ≥ budget
   ↓
ReplyDraft pipeline (P1.5: research → drafts → Telegram review → post)
```

Each guard returns `Option<SkipReason>`; `None` means proceed. The first guard that returns `Some(reason)` short-circuits and the mention is recorded as `skipped: <reason>` in the existing `MentionStore`.

## 4. ThreadDepthGuard (P1.7-A)

### What it catches

The dominant AI-to-AI loop shape: another bot replies to your reply. Without this guard, heartbit polls mentions, sees the bot's reply (which threads on your tweet), drafts a counter-reply, and the loop accelerates by your `[Pick]` count.

### How it works

Pure logic. Reads `mention.in_reply_to_tweet_id` (already on the `Mention` struct from P1.5). If `Some(parent_id)` AND `MentionStore::was_replied(&parent_id).await? == true` → `Some(SkipReason::OwnThreadContinuation)`.

### Configuration

```toml
[[daemon.persona_mentions]]
# ... existing fields ...
enable_thread_depth_guard = true   # default true
```

When disabled, the guard returns `None` always.

### Cost

Zero LLM calls. One `MentionStore::was_replied` lookup per mention.

## 5. BotHeuristicGuard (P1.7-B)

### What it catches

Mentions from automated accounts. Conservative — needs ≥ 2 of 3 signals to fire (single signal is too noisy for auto-skip, given the cost of a false positive is annoying a real human).

### Three signals

#### Signal 1: Handle suffix/prefix match

Configurable list of substrings that strongly suggest a bot. Case-insensitive substring match on the author's handle:

```toml
suspicious_handle_patterns = ["_bot", "_gpt", "_ai", "ai_", "gpt_", "bot_"]
```

Defaults to the list above. Matches `chatgpt_bot`, `claude_ai`, `gpt_responder`, etc. Strict substring (not regex) to keep cheap and predictable.

#### Signal 2: Follower/following ratio anomaly

Bots typically follow many accounts to attract follow-backs but gain few real followers. Skip when:

```
followers / following < min_follower_following_ratio
```

Default `0.05`. Disabled when `following_count` is `None` or `0`. Disabled when `follower_count` is unavailable.

This requires fetching the X user object's `following_count` field. The existing P1.5 `MentionerContext::recent_tweets` fetch can include it via `user.fields=public_metrics` in the same call — no extra HTTP round-trip.

#### Signal 3: Account age

Brand-new accounts that immediately mention many users are highly suspect. Skip when:

```
account.created_at older than `min_account_age_days` is FALSE
```

Default `7`. Requires `user.fields=created_at` on the user fetch (same call, additive).

### Threshold

```toml
bot_heuristic_threshold = 2   # default 2 (out of 3 signals)
```

Skip only when at least `threshold` signals match. Setting to `0` disables; setting to `1` makes any signal fire (aggressive); setting to `3` requires all (very conservative).

### Configuration

```toml
[[daemon.persona_mentions]]
enable_bot_heuristic_guard = true   # default true
suspicious_handle_patterns = [...]   # default as above
min_follower_following_ratio = 0.05
min_account_age_days = 7
bot_heuristic_threshold = 2
```

### Skip reason

`SkipReason::BotSuspected { reasons: Vec<String> }` — captures which signals matched, for the audit log.

### Cost

Zero LLM calls. One additional `user.fields` query parameter on the existing X user fetch (no extra round-trip).

## 6. ConversationDepthGuard (P1.7-C)

### What it catches

Even when a third party (not your direct correspondent) joins a thread, you can still get sucked into 5-, 10-, 20-message back-and-forths. This guard caps reply count per X conversation regardless of who participates.

### Data model change

`Mention` gains a new field: `conversation_id: Option<String>`. The X API returns this when `tweet.fields=conversation_id` is requested on the mentions GET. P1.5's `TwitterMentionsTool` already uses `tweet.fields=author_id,created_at,in_reply_to_user_id` — extend to `author_id,created_at,in_reply_to_user_id,conversation_id`.

### MentionStore extensions

Two new methods on the trait:

```rust
fn replies_in_conversation<'a>(
    &'a self,
    conversation_id: &'a str,
) -> Pin<Box<dyn Future<Output = Result<usize, StoreError>> + Send + 'a>>;

fn record_reply_in_conversation<'a>(
    &'a self,
    conversation_id: &'a str,
) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>>;
```

Both `InMemoryMentionStore` and `JsonlMentionStore` implementations are extended. The JSONL impl gains a `ConversationReply { conversation_id }` event.

The increment happens AFTER a successful `Posted` outcome from the reply pipeline (callback through `handle_reply_draft`). Skipped/TimedOut/etc. do NOT count against the conversation cap, since no reply was actually posted.

### Skip rule

If `mention.conversation_id.is_some()` AND `replies_in_conversation(id) >= per_conversation_max_replies` → `Some(SkipReason::ConversationDepthExceeded { conversation_id, count, cap })`.

### Configuration

```toml
[[daemon.persona_mentions]]
per_conversation_max_replies = 2   # default 2; set to 0 to disable
```

### Edge cases

- **Mention has no conversation_id** (rare; X API normally returns it): guard returns `None` (proceed). Logged at debug level.
- **First reply to a conversation**: `replies_in_conversation` returns 0; passes the guard.
- **Self-replies** (already filtered by `SpamGuard::SelfReply`): never reach this guard.
- **Conversation-id ≠ thread root**: X's `conversation_id` is the root tweet of the thread tree. Multiple branches under the same root share the conversation_id. Per-conversation cap is intentionally global across branches — it caps total noise per logical thread.

### Cost

Zero LLM calls. Two `MentionStore` lookups per mention (existing `was_replied` + new `replies_in_conversation`).

## 7. DailyBudgetGuard (P1.7-D)

### What it caps

Total LLM tokens spent by the mentions pipeline per persona per UTC day. Includes the ReplyDraft pipeline's full cost (research + writer ×N + critic + fact + judge + image_generator if used).

### Data model

New `DailyTokenBudget` trait + `JsonlDailyBudget` impl, mirroring `MentionStore`'s shape:

```rust
pub trait DailyTokenBudget: Send + Sync {
    /// Total tokens recorded for `persona` on `date` (YYYY-MM-DD UTC).
    fn usage_today<'a>(
        &'a self,
        persona: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<u64, BudgetError>> + Send + 'a>>;

    /// Add `tokens` to today's accumulator for `persona`.
    fn record_usage<'a>(
        &'a self,
        persona: &'a str,
        tokens: u64,
    ) -> Pin<Box<dyn Future<Output = Result<(), BudgetError>> + Send + 'a>>;
}
```

`JsonlDailyBudget` keeps an in-memory `HashMap<(String, NaiveDate), u64>`, replays from a JSONL file at open. Each `record_usage` appends a line `{"date": "2026-05-10", "persona": "heartbit-ghost:x", "tokens": 1234}` and updates the in-memory mirror. Reset is implicit: the date key changes at UTC midnight.

### Skip rule

Before invoking the reply pipeline, check `usage_today(persona) >= budget`. If yes → `Some(SkipReason::DailyBudgetExhausted { used, budget })`.

After every reply pipeline run (regardless of outcome — even Telegram-skipped runs already paid the LLM cost), call `record_usage(persona, output.usage_summary.input_tokens + output_tokens + reasoning_tokens)`.

### Configuration

```toml
[[daemon.persona_mentions]]
daily_token_budget = 50000   # default None (= unlimited)
budget_store = "jsonl"        # default "in_memory"
budget_path = "~/.heartbit/ghost/budgets/heartbit-ghost-x.jsonl"
```

When `daily_token_budget = None` (default), the guard always returns `None`. The JSONL store is still created if `budget_store = "jsonl"` for audit purposes.

### Telegram side-effect

When the budget transitions from "below cap" → "at cap" within a tick (i.e., the previous tick had headroom and this one doesn't), the daemon logs `tracing::warn!(persona, used, budget, "daily LLM budget exhausted; replies suspended until UTC midnight")`. Subsequent ticks within the same UTC day skip silently — no Telegram notification (per user choice during brainstorming).

### Cost reset

Implicit at UTC midnight. The next call to `usage_today(persona)` returns 0 because the date key changed. No explicit reset job needed.

### Cost

Zero LLM calls. One `usage_today` read per mention (cheap, in-memory). One `record_usage` write per pipeline run.

## 8. Skip reason taxonomy

`SkipReason` (extends the P1.5 enum):

```rust
pub enum SkipReason {
    // P1.5 (existing)
    SelfReply,
    StaleParent,
    LowEffortSpam,
    PerAuthorRateLimit,
    TooShortToEngage,
    // P1.7 (new)
    OwnThreadContinuation,
    BotSuspected { reasons: Vec<String> },
    ConversationDepthExceeded { conversation_id: String, count: usize, cap: usize },
    DailyBudgetExhausted { used: u64, budget: u64 },
}
```

The handler logs a single `tracing::info!` per skip with the reason. The mention is recorded in `MentionStore.replied` with the skip reason so the same mention isn't reconsidered on the next poll.

## 9. Files

### New files

- `crates/heartbit-ghost/src/reply/thread_guard.rs` — `ThreadDepthGuard` (~70 LOC + tests)
- `crates/heartbit-ghost/src/reply/bot_guard.rs` — `BotHeuristicGuard` + `BotHeuristicConfig` (~120 LOC + tests)
- `crates/heartbit-ghost/src/reply/conversation_guard.rs` — `ConversationDepthGuard` (~50 LOC + tests)
- `crates/heartbit-ghost/src/reply/budget.rs` — `DailyTokenBudget` trait + `InMemoryDailyBudget` + `JsonlDailyBudget` (~200 LOC + tests)
- `crates/heartbit-ghost/src/reply/budget_guard.rs` — `DailyBudgetGuard` thin wrapper (~40 LOC + tests)

### Modified files

- `crates/heartbit-ghost/src/reply/storage.rs` — add `replies_in_conversation` + `record_reply_in_conversation` methods to `MentionStore` trait + both impls.
- `crates/heartbit-ghost/src/reply/mod.rs` — add `conversation_id: Option<String>` field to `Mention` struct (with `#[serde(default)]` for backward compat with P1.5 stores). Add new variants to `SkipReason` enum.
- `crates/heartbit-ghost/src/tools/mentions.rs` — extend `tweet.fields` query param to include `conversation_id`. Map into the returned `Mention.conversation_id`.
- `crates/heartbit-ghost/src/reply/spam_guard.rs` — re-export `SkipReason` extensions if needed.
- `crates/heartbit/src/daemon/mention_poll_handler.rs` — extend `handle_mention_poll` to evaluate the 4 new guards in order BEFORE dispatching `ReplyDraft`. Increment conversation counter after each successful `Posted` outcome (callback or post-pipeline).
- `crates/heartbit/src/daemon/mention_context.rs` — add `bot_heuristic_config`, `per_conversation_max_replies`, `daily_token_budget`, `budget_store: Arc<dyn DailyTokenBudget>` fields to `PersonaMentionEntry`.
- `crates/heartbit-core/src/config/daemon.rs` — extend `PersonaMentionsConfig` with the new fields (all optional with sensible defaults).
- `crates/heartbit-cli/src/daemon/mod.rs` — pass-through wiring for the new config fields into `PersonaMentionEntry`.

## 10. Test plan

Unit tests (TDD throughout):

- `ThreadDepthGuard`:
  - `skips_when_parent_in_replied_set`
  - `proceeds_when_parent_not_in_replied_set`
  - `proceeds_when_no_parent_id`
  - `disabled_via_config_returns_none`
- `BotHeuristicGuard`:
  - `handle_pattern_signal_matches_substring_case_insensitive`
  - `follow_ratio_signal_matches_below_threshold`
  - `account_age_signal_matches_recent_account`
  - `threshold_2_requires_two_signals`
  - `single_signal_does_not_skip_at_threshold_2`
  - `disabled_via_threshold_0`
- `ConversationDepthGuard`:
  - `skips_at_cap`
  - `proceeds_below_cap`
  - `proceeds_when_conversation_id_absent`
  - `cap_zero_disables`
- `DailyBudgetGuard`:
  - `proceeds_when_below_budget`
  - `skips_when_at_or_above_budget`
  - `none_budget_always_proceeds`
  - `usage_resets_at_utc_midnight` (uses fixed-time fixture)
- `JsonlDailyBudget`:
  - `record_then_read_round_trip`
  - `reload_from_disk_preserves_today_only` (i.e., yesterday's records are still in the file but `usage_today` returns 0 for the new day)
  - `per_persona_isolation`
  - `missing_file_returns_zero`
- `MentionStore` extensions:
  - `record_then_count_in_conversation_round_trip`
  - `count_zero_for_unknown_conversation`
- `TwitterMentionsTool`:
  - `mentions_response_includes_conversation_id_when_field_requested`

Integration test:
- `mention_poll_with_4_guards_filters_mixed_fixture` — fixture with 6 mentions:
  - 1 normal (passes all guards) → dispatched
  - 1 own-thread continuation (caught by ThreadDepth) → skipped
  - 1 bot-suspected (caught by BotHeuristic) → skipped
  - 1 conversation-cap-exceeded (caught by ConversationDepth) → skipped
  - 1 budget-exhausted state (caught by DailyBudget) → skipped
  - 1 already-handled-by-P1.5-SpamGuard (e.g., self-reply) → skipped
  
  Asserts: only 1 `ReplyDraft` command produced; 5 distinct skip reasons in logs/store.

Acceptance:
- Quality gate (`cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace --features daemon`)
- Live test: configure `[[daemon.persona_mentions]]` with `daily_token_budget = 5000` (low for testing). Reply to a few mentions until the budget exhausts. Verify the daemon logs the warn and stops drafting until UTC midnight.

## 11. Hors scope (explicit)

- ❌ ML-based bot detection — heuristics only for V1
- ❌ Cross-persona budget aggregation — each persona has its own budget
- ❌ Adaptive cost throttling — no "slow down at 80%, stop at 100%"; just hard cap
- ❌ Refunding budget on Telegram-skip — Skipped runs still count their token cost
- ❌ Auto-resume notification at midnight — silent reset, operator re-discovers via the next dispatched ReplyDraft
- ❌ Whitelist/blacklist of specific authors that bypass the bot heuristic — operator can disable the heuristic globally if needed
- ❌ Replies-from-bots-only mode (auto-pick instead of Telegram review for known-trusted bots) — still calibration mode regardless

## 12. Risks

| Risk | Mitigation |
|------|-----------|
| BotHeuristic false-positive skips a real human | Conservative threshold (≥2 of 3 signals); operator can disable per persona; the existing Telegram calibration gate is the ultimate human-in-the-loop |
| ThreadDepthGuard skips legitimate continuation (e.g., a real follow-up question from a real person) | Documented limitation; operator can disable the guard if their use case requires reply chains. Per-author rate limit (P1.5) still applies |
| ConversationDepth cap of 2 is too low | Configurable; default `2` aligns with "ack + one follow-up = enough". Can be raised to 5 or higher |
| DailyBudgetGuard's `record_usage` is racy across concurrent handler invocations | The handler is invoked serially per persona by the cron scheduler; no concurrency on a single persona. Cross-persona writes are safe (different keys). Use `tokio::sync::RwLock` on the in-memory mirror |
| Budget JSONL grows unbounded | Each persona writes ~1 line per reply pipeline run; even at heavy usage (~50/day), file growth is ~5 KB/year. Acceptable. Can add log rotation in P1.8+ |
| `conversation_id` field in MentionStore breaks deserialization of old JSONL stores | `#[serde(default)]` on the new field handles missing values; old entries deserialize cleanly with `conversation_id: None` |
| Bot heuristic over-triggers on legitimate creator accounts (e.g., `the_ai_dude`) | Configurable suspicious patterns; operator can override defaults. Threshold of 2 means a single substring match alone won't fire |

## 13. Sub-phases

- **P1.7a** — `ThreadDepthGuard` + `BotHeuristicGuard` (pure-logic guards) + `Mention.conversation_id` plumbing (Tasks 1-5)
- **P1.7b** — `MentionStore` extensions + `ConversationDepthGuard` (Tasks 6-8)
- **P1.7c** — `DailyTokenBudget` trait + 2 impls + `DailyBudgetGuard` (Tasks 9-11)
- **P1.7d** — Integration into `handle_mention_poll` + acceptance (Tasks 12-13)

Each sub-phase ships with passing tests at the end. Total: ~13 tasks, similar size to P1.5/P1.6.

## 14. Open questions

None at design freeze. All judgment calls resolved during brainstorming:
- 4 guards (all enabled in scope) ✅
- Bot heuristic threshold default = 2 of 3 signals ✅
- Conversation cap default = 2 replies ✅
- Daily budget default = None (unlimited; operator must opt in) ✅
- Budget exhaust behavior = skip + log warn (no Telegram alert in V1) ✅
- Reset = midnight UTC (date-keyed; implicit) ✅

Future iterations (out of P1.7 scope) could add: ML-based bot detection, adaptive cost throttling, per-conversation budgets, scoped whitelist/blacklist for trusted accounts.
