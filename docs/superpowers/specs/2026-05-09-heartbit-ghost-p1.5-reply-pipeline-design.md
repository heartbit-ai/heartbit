# heartbit-ghost P1.5 — Reply pipeline (mention polling + reply drafting + Telegram review)

**Status:** awaiting approval (2026-05-09)
**Predecessor:** heartbit-rs:x persona + image-prompt engraving fix merged to `main` (commits `784b8e5` + `dc05364`)
**Branch:** `feat/heartbit-ghost-p1.5-reply-pipeline` (created off `main`)
**Brainstorming:** done inline 2026-05-09 (this conversation, before this doc)

---

## 1. Goal

Close the feedback loop on existing tweets: poll for new mentions / replies on the operator's account, draft a tailored response per mention via the persona's voice, route each draft to Telegram for review, and post the chosen reply via the existing `twitter_reply` tool.

The umbrella spec (`2026-05-07-heartbit-ghost-x-agent-design.md` §12 acceptance) listed this loop as v0.1 acceptance criteria but it was never built — only the bare tools (`TwitterMentionsTool`, `TwitterReplyTool`) shipped in P1.1. This phase wires them together.

After P1.5 the operator can:
- Receive a Telegram review (parent tweet + mention + 1-2 candidate replies + Skip/Pick buttons) within ~5 minutes of a new mention
- Pick a draft → reply posts under the mention
- Skip → mention is marked seen, no reply
- The persona's voice (currently `heartbit-ghost:x` or `heartbit-rs:x`) drives the reply's tone and constraints

This is **calibration mode only** for V1: every reply goes through Telegram. Auto-reply is a P1.4 (autonomy) concern, not P1.5.

## 2. Architecture

```
CronScheduler (every 5 min, configurable)
  ↓ submits MentionPoll command per active persona
MentionPoll task:
  - read since_id from store
  - call twitter_mentions(user_id, since_id) → N new mentions
  - for each mention NOT in replied_to set, run anti-spam guards
  - for each surviving mention, dispatch SubmitTask(ReplyDraft { mention, persona })
  - update since_id to max(mention.id) seen
ReplyDraft task (one per mention):
  ↓ run_reply_pipeline:
    mini-researcher (fetch parent tweet via twitter_search/get + author bio
                     via twitter_user) → digest
    ↓
    reply_writer (NEW recipe: ≤280 chars, voice-conditioned, addresses
                  mention's specific content)
    ↓
    style_critic + fact_check (existing recipes)
    ↓
    publish_gate (cap = 280 for replies, no thread split)
    ↓
    Telegram review delivery (NEW message shape: parent context + mention
                              + 1-2 drafts + Skip/Pick)
  ↓ on user pick:
twitter_reply(text, in_reply_to=mention.id) → posted
  ↓ on Skip / TimedOut:
mark mention.id as seen (no post)
  ↓ either way:
add mention.id to replied_to set
```

Two new daemon-side tasks: `MentionPoll` (cron-driven, fan-out) and `ReplyDraft` (per-mention, fan-in). Both use the existing `CommandProducer` mechanism — no new transport.

### 2.1 Why two distinct pipelines (`run_pipeline` vs. `run_reply_pipeline`)?

The original-thread pipeline generates a thread from a topic. The reply pipeline generates a 280-char reply to an existing tweet. The differences are large enough that sharing a single function would muddy both:

| Concern | run_pipeline | run_reply_pipeline |
|---|---|---|
| Output | 1-12 tweets, possibly with image | 1 tweet, no image |
| Length cap | per-tweet 280; thread length governed by `thread_max_length` | hard 280, no thread |
| Researcher input | free-form topic | mention text + parent tweet + author context |
| Multi-candidate | 3 candidates by default + judge | 1-2 candidates, judge optional |
| Image generation | yes (head tweet) | no |
| Telegram review | thread-shape buttons | reply-shape buttons (parent quoted) |
| `mode_addendum` | applies | applies (same persona voice) |

A separate function keeps each call site readable. Shared sub-agent recipes (`style_critic`, `fact_check`) are reused.

## 3. Files

| Path | Action | Purpose |
|------|--------|---------|
| `crates/heartbit-ghost/src/agents/reply_writer.rs` | NEW | `reply_writer_recipe()` — system prompt for short, mention-anchored, persona-voiced replies. ~3 unit tests. |
| `crates/heartbit-ghost/src/agents/mod.rs` | MODIFY | Re-export `reply_writer_recipe`. |
| `crates/heartbit-ghost/src/reply/mod.rs` | NEW | `run_reply_pipeline`, `ReplyConfig`, `ReplyOutput`, `ReplyOutcome`, `ReplyError`. ~6 integration tests using existing `MockProvider::route_with_recorder`. |
| `crates/heartbit-ghost/src/reply/prompts.rs` | NEW | `build_reply_research_user_message`, `build_reply_writer_user_message`, `build_reply_critic_user_message`, `build_reply_fact_user_message`. ~3 unit tests. |
| `crates/heartbit-ghost/src/reply/delivery.rs` | NEW | `ReplyReviewDelivery` trait + `ReplyReviewMessage` shape. Mirrors `review/delivery.rs` but with parent-tweet context. |
| `crates/heartbit-ghost/src/reply/storage.rs` | NEW | `MentionStore` trait: `since_id_for(persona, user_id)`, `mark_replied(mention_id)`, `was_replied(mention_id)`, `bump_since_id(...)`. `InMemoryMentionStore` (default) + `JsonlMentionStore` (persistent). |
| `crates/heartbit-ghost/src/reply/spam_guard.rs` | NEW | `SpamGuard::should_skip(mention) -> Option<SkipReason>`. Encodes the 4 anti-spam rules from §8. |
| `crates/heartbit-ghost/src/lib.rs` | MODIFY | `pub mod reply;`. |
| `crates/heartbit/src/daemon/commands.rs` (or wherever `DaemonCommand` lives) | MODIFY | Add `DaemonCommand::MentionPoll { persona, user_id }` and `DaemonCommand::ReplyDraft { persona, mention, parent_context }`. |
| `crates/heartbit/src/daemon/dispatcher.rs` | MODIFY | Dispatch arms for the two new commands. |
| `crates/heartbit-cli/src/persona.rs` | MODIFY | New subcommand `persona reply once <NAME> --mention-id <ID>` for manual one-off testing without the daemon. |
| `crates/heartbit-telegram/src/delivery.rs` (existing) | MODIFY | Implement `ReplyReviewDelivery` for `TelegramReviewDelivery` (variant message shape). |

Plus 2 new daemon-config knobs:

```toml
[daemon.persona.heartbit-ghost:x.mentions]
enabled = true
poll_interval_seconds = 300       # 5 min default per spec §11
user_id = "1234567890"             # operator's X user id (no auto-resolve from handle in V1)
candidates_per_reply = 2           # 1 = no judge; 2 = judge picks; >2 rejected
mention_store = "jsonl"            # "in_memory" or "jsonl"
mention_store_path = "~/.heartbit/ghost/mentions/heartbit-ghost:x.jsonl"
```

## 4. The `reply_writer` recipe

### 4.1 System prompt (full text)

```
You write a single short reply (≤280 characters) to a specific tweet. The reply must address the content of that tweet directly — never a generic acknowledgement, never a content-free thanks, never a question that the tweet's author obviously already considered.

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
```

### 4.2 Recipe shape

```rust
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
```

`max_turns: Some(1)` — replies don't need iteration. `max_tokens: Some(512)` — 280 chars + small headroom for revision turns the writer can never actually use given `max_turns: 1`. `reasoning_effort: "low"` — replies are not the place for elaborate planning; the writer riffs from the input.

## 5. `run_reply_pipeline`

### 5.1 ReplyConfig

```rust
#[derive(Clone)]
pub struct ReplyConfig<'a> {
    pub persona_name: &'a str,
    pub provider: Arc<BoxedProvider>,
    pub corpora_root: &'a Path,
    pub profiles_root: &'a Path,
    pub on_progress: Option<ProgressCallback>,
    pub mention: Mention,                    // see 5.2
    pub parent: Option<TweetSnapshot>,        // operator's tweet the mention replied to
    pub mentioner_context: Option<MentionerContext>,  // bio + 2-3 recent tweets
    pub candidates_per_reply: usize,          // 1..=3, default 2
    pub mode_addendum: Option<&'a str>,
    pub researcher_override: Option<ResearcherOverride>,  // typically None for replies
    pub delivery: Arc<dyn ReplyReviewDelivery>,
    pub twitter_tool: Arc<dyn Tool>,
    pub credentials: Arc<dyn CredentialResolver>,
}
```

### 5.2 Mention + supporting types

```rust
#[derive(Debug, Clone)]
pub struct Mention {
    pub id: String,
    pub text: String,
    pub author_id: String,
    pub author_handle: String,
    pub posted_at: chrono::DateTime<chrono::Utc>,
    /// The tweet this mention is replying to (None if it's a top-level @-mention).
    pub in_reply_to_tweet_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TweetSnapshot {
    pub id: String,
    pub text: String,
    pub posted_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct MentionerContext {
    pub handle: String,
    pub bio: Option<String>,
    pub recent_tweets: Vec<TweetSnapshot>,  // up to 3
    pub follower_count: Option<u64>,
}
```

### 5.3 Pipeline flow

```rust
pub async fn run_reply_pipeline(cfg: ReplyConfig<'_>) -> Result<ReplyOutput, ReplyError> {
    // 1. Validate cfg.candidates_per_reply ∈ 1..=3.
    // 2. Load profile snapshot (same as run_pipeline).
    // 3. Build mini-researcher (writer + critic + fact runners share this provider).
    //    The mini-researcher's user message is the mention + parent + mentioner_context.
    //    Output: a 1-3 sentence digest naming the SPECIFIC point to engage with.
    // 4. Generate N reply candidates via reply_writer in parallel (tokio::JoinSet,
    //    same as pipeline/mod.rs).
    // 5. style_critic on each; reject candidates with "no_reply" output upstream.
    // 6. fact_check on the survivor(s).
    // 7. If candidates > 1, run judge to pick one; else use the single survivor.
    // 8. publish_gate: hard 280-char cap; no thread split.
    // 9. Build ReplyReviewMessage (parent-tweet quoted + mention + chosen reply).
    // 10. Deliver via cfg.delivery → wait for user pick.
    // 11. On Pick: cfg.twitter_tool.execute(twitter_reply input) → posted.
    //     On Skip / TimedOut: no post, but mention is marked seen.
}
```

### 5.4 ReplyOutput

```rust
#[derive(Debug, Clone)]
pub struct ReplyOutput {
    pub mention_id: String,
    pub candidates: Vec<ReplyCandidateRecord>,
    pub usage_summary: TokenUsage,
    pub outcome: ReplyOutcome,
}

#[derive(Debug, Clone)]
pub enum ReplyOutcome {
    Posted { reply_tweet_id: String, reply_url: String, chosen_index: usize },
    Skipped,
    TimedOut,
    GateRejected { chosen_index: usize, reason: String },
    PublishFailed { chosen_index: usize, reason: String },
    NoReply,  // writer chose to not reply (all candidates returned "no_reply")
}
```

## 6. Mention polling task

### 6.1 New `DaemonCommand` variants

```rust
DaemonCommand::MentionPoll {
    persona: String,
    user_id: String,
}

DaemonCommand::ReplyDraft {
    persona: String,
    mention: Mention,
    parent: Option<TweetSnapshot>,
    mentioner_context: Option<MentionerContext>,
}
```

The `MentionPoll` command is fired by the cron scheduler. The `ReplyDraft` command is fired by the `MentionPoll` handler for each surviving mention (post-spam-guard). Both commands flow through the same `DaemonCommand` channel — no new transport.

### 6.2 `MentionPoll` handler

```rust
// Pseudocode in dispatcher
async fn handle_mention_poll(persona: &str, user_id: &str, store: &dyn MentionStore, ...) {
    let since_id = store.since_id_for(persona, user_id).unwrap_or(None);
    let resp = twitter_mentions.execute(MentionsInput {
        user_id: user_id.to_string(),
        max_results: 50,
        since_id,
    }).await?;
    let mentions: Vec<Mention> = resp.mentions;  // see tools/mentions.rs

    let mut max_seen_id: Option<String> = since_id.clone();
    for m in mentions.into_iter() {
        // Anti-double-reply
        if store.was_replied(&m.id).await? { continue; }
        // Anti-spam guards
        if let Some(reason) = spam_guard.should_skip(&m, ...) {
            store.mark_replied(&m.id).await?;  // mark seen so we don't retry
            log::info!("spam-skip mention {}: {:?}", m.id, reason);
            continue;
        }
        // Fetch parent tweet + mentioner context (cheap; same X API tier)
        let parent = if let Some(pid) = m.in_reply_to_tweet_id.clone() {
            twitter_get_tweet(pid).await.ok()
        } else { None };
        let mentioner_context = build_mentioner_context(&m.author_id).await.ok();
        // Dispatch reply draft
        producer.send(DaemonCommand::ReplyDraft {
            persona: persona.to_string(),
            mention: m.clone(),
            parent,
            mentioner_context,
        }).await?;
        if max_seen_id.as_deref().map_or(true, |s| m.id.as_str() > s) {
            max_seen_id = Some(m.id);
        }
    }
    if let Some(new_since) = max_seen_id {
        store.bump_since_id(persona, user_id, &new_since).await?;
    }
}
```

`since_id` advancement is monotonic — it tracks the LATEST mention seen, regardless of whether each individual one was replied to.

### 6.3 `ReplyDraft` handler

Constructs `ReplyConfig` from the daemon's persona registry + the command payload, calls `run_reply_pipeline`, and persists the outcome to the audit log (same store the main pipeline uses). On `Posted` or `Skipped` or `TimedOut`, marks the mention as replied via `store.mark_replied(...)`.

## 7. Mention storage

### 7.1 `MentionStore` trait

```rust
#[async_trait::async_trait]
pub trait MentionStore: Send + Sync {
    async fn since_id_for(&self, persona: &str, user_id: &str) -> Result<Option<String>, StoreError>;
    async fn bump_since_id(&self, persona: &str, user_id: &str, new_id: &str) -> Result<(), StoreError>;
    async fn mark_replied(&self, mention_id: &str) -> Result<(), StoreError>;
    async fn was_replied(&self, mention_id: &str) -> Result<bool, StoreError>;
}
```

### 7.2 Implementations

- **`InMemoryMentionStore`** — `Arc<RwLock<...>>` of two maps. Default. Loses state across daemon restarts; OK for local development but loses replied_to history (which would cause double-reply on restart for the brief window between last poll and the restart).
- **`JsonlMentionStore`** — Append-only JSONL on disk. One line per event: `{"event": "since_id", "persona": "...", "user_id": "...", "id": "..."}` or `{"event": "replied", "mention_id": "...", "ts": "..."}`. Loaded into memory at startup; new events appended. Compaction is out of scope for V1 (jsonl can grow; V1 ships a manual compaction script in the plan).

Postgres-backed store is **out of scope for P1.5** — V1 uses JSONL. P1.4 (which adds the full audit log to Postgres) can fold this into the Postgres schema later.

## 8. Anti-spam guards

`SpamGuard::should_skip(&mention, &context) -> Option<SkipReason>` — returns `Some(reason)` to skip, `None` to proceed.

Rules (in evaluation order, fail-fast on first match):

1. **Self-reply**: `mention.author_id == operator_user_id` → skip (`SkipReason::SelfReply`)
2. **Stale parent**: `parent.posted_at < now - 7 days` → skip (`SkipReason::StaleParent`). Anti-thread-necromancy.
3. **Low-follower spam**: `mentioner_context.follower_count < 5` AND `mention.text.len() < 30` → skip (`SkipReason::LowEffortSpam`). Both signals together; either alone is a false-positive risk.
4. **Per-author rate limit**: replied to this `author_id` >= 3 times in the last 24 hours → skip (`SkipReason::PerAuthorRateLimit`). Tracked via `MentionStore`.
5. **Empty / single-emoji**: `mention.text.trim().chars().filter(|c| c.is_alphanumeric()).count() < 3` → skip (`SkipReason::TooShortToEngage`).

All thresholds are configurable via daemon TOML (with defaults above). The spam-skipped mentions are still marked replied so they don't retry.

A `SkipReason::HostileContent` rule is **deferred to P1.4** — content-classification guardrails belong in the same phase as the always-on PII/defamation/harassment guardrails (umbrella spec §7.3). For P1.5 we trust the `reply_writer`'s own "if it's bait, output no_reply" instruction.

## 9. Telegram delivery adapter

### 9.1 `ReplyReviewMessage` shape

The Telegram message for a reply review needs to show the operator three pieces of information they can scan in 5 seconds:

```
[bot]
NEW MENTION on your tweet from @<mentioner_handle> (<follower_count> followers)

YOUR TWEET (parent):
> <parent.text first 200 chars + ellipsis if truncated>

THEIR REPLY:
> <mention.text>

DRAFT 1:
> <candidate 1>

DRAFT 2:    (only when candidates_per_reply > 1)
> <candidate 2>

[ 1 ] [ 2 ] [ Skip ]
```

The buttons are `1`, `2`, `Skip`. (No `3` because we cap at 2 candidates for replies — judge picks between them when 2.)

### 9.2 `ReplyReviewDelivery` trait

Mirrors `review::ReviewDelivery` (deliver + report) but with the reply-specific message body. Same `DeliveryReceipt` / `DeliveryOutcome` shape so the existing receipt-handling code in `heartbit-cli` is reused.

```rust
#[async_trait::async_trait]
pub trait ReplyReviewDelivery: Send + Sync {
    async fn deliver(&self, msg: ReplyReviewMessage) -> Result<DeliveredReplyReview, ReviewDeliveryError>;
    async fn report(&self, receipt: DeliveryReceipt, outcome: ReplyOutcome) -> Result<(), ReviewDeliveryError>;
}
```

The existing `TelegramReviewDelivery` in `heartbit-cli` gains a second `impl ReplyReviewDelivery` block.

## 10. Tests

| Layer | Count | What it covers |
|-------|-------|----------------|
| `reply_writer` unit | 3 | recipe shape, system prompt mandates ≤280, prompt forbids generic openers |
| `reply::prompts` unit | 3 | mention quoted in researcher message, parent-context optional, mode_addendum appended |
| `reply::storage` unit | 5 | InMemoryMentionStore CRUD; JsonlMentionStore round-trip; was_replied on missing returns false; bump_since_id is monotonic; per-author count |
| `reply::spam_guard` unit | 5 | one per rule (self-reply, stale-parent, low-follower-spam, per-author-rate, too-short) |
| `run_reply_pipeline` integration | 6 | happy path with 2 candidates → posts; "no_reply" → outcome NoReply; publish_gate rejects 281-char draft; user Skip → outcome Skipped; user TimedOut → outcome TimedOut; twitter_reply API error → PublishFailed |
| Daemon `MentionPoll` integration | 2 | poll with no new mentions does nothing; poll with 3 new mentions dispatches 3 ReplyDraft commands AND bumps since_id |
| Live (manual) | 1 | end-to-end: post a tweet, have a colleague reply, watch Telegram, pick, verify reply lands under the colleague's mention |

Total: ~24 new automated tests + 1 manual live test.

## 11. Out of scope (explicitly deferred)

- **Auto-reply (no Telegram review)** — P1.4 territory. Calibration mode only for V1.
- **Image attachments to replies** — possible but not in scope; replies are short and text-first.
- **Quote-tweet replies** vs. plain replies — V1 always emits plain replies. Quote-tweet is a different X API call (`POST /2/tweets` with `quote_tweet_id`) and is a future enhancement.
- **Postgres-backed `MentionStore`** — V1 ships JSONL; Postgres folds in via P1.4 audit log.
- **Reply chain awareness** — V1 treats each mention as independent. Threading across multiple back-and-forths ("conversational mode") is deferred.
- **Sentiment / tone classification beyond the writer's own self-restraint** — the `reply_writer` is told to detect bait and emit `no_reply`; explicit content classifiers come in P1.4.
- **`heartbit persona reply once` CLI subcommand** — actually IN scope (§3) for one-off testing. But not a full operator workflow.
- **Multi-account mention polling** — V1 polls one `user_id` per persona. Multi-account is a multi-tenancy concern.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Replying to a stale mention creates an awkward thread necromancy | Stale-parent guard (rule 2 in §8) skips parents older than 7 days. |
| Replying to spam / low-effort accounts dilutes voice | Low-follower + short-text guard; per-author rate limit; reply_writer's own "no_reply" escape hatch. |
| Per-author over-engagement (replying 10× in a day to the same person) | Per-author rate limit: max 3 replies per 24h. |
| Operator misses a Telegram review and posts go stale | Per ReviewDelivery's existing `interaction_timeout_seconds`, `TimedOut` is a recognized outcome and the mention is marked seen — won't be re-drafted on the next poll. |
| `since_id` corruption on disk causes mass re-reply | JsonlMentionStore loads all events at startup AND maintains the `replied_to` set; even if `since_id` is wrong, `was_replied` blocks the actual reply. Belt + suspenders. |
| `reply_writer` produces obsequious / generic replies despite the prompt | Prompt explicitly bans "Thanks for…" / "Great point…" openers; style_critic also evaluates against the persona's `ai_tells_to_avoid` list. |
| Cron poll fires while a previous ReplyDraft is still running | `DaemonCommand` is queued via the existing producer; the dispatcher processes them in sequence. The cron interval (5 min default) is much longer than typical pipeline runtime (~30s). No need for explicit deduplication. |
| Two operators sharing one Telegram chat see the same review | Out of scope; same as the existing review pipeline. |

## 13. Implementation phases

P1.5 ships in three sub-phases. Each phase is self-testable; sub-phases B and C depend on A.

### P1.5a — Pipeline + writer recipe (3-4 days)

- `reply_writer_recipe()` + system prompt
- `run_reply_pipeline` with all 6 stages (research / write / critic / fact / judge / gate)
- `ReplyConfig`, `ReplyOutput`, `ReplyOutcome`, `ReplyError`, `Mention`, `TweetSnapshot`, `MentionerContext`
- Unit tests for the recipe + prompts
- Integration tests for the pipeline using `MockProvider::route_with_recorder` + a `MockReplyReviewDelivery` + a `MockTwitterReplyTool`
- A `heartbit persona reply once <NAME> --mention-id <ID>` CLI subcommand for one-off testing without the daemon

After P1.5a, an operator can manually trigger a reply draft from a known mention ID and see it through Telegram review.

### P1.5b — Storage + spam guards (3-4 days)

- `MentionStore` trait + `InMemoryMentionStore` + `JsonlMentionStore`
- `SpamGuard` with 5 rules + configurable thresholds
- Unit tests per rule + storage round-trip tests
- Integration test that wires `JsonlMentionStore` into `run_reply_pipeline` (storage hooks at the dispatcher level, not inside the pipeline)

After P1.5b, the storage layer + guards are testable in isolation. Still no daemon polling.

### P1.5c — Daemon polling + Telegram delivery (2-3 days)

- `DaemonCommand::MentionPoll` + `DaemonCommand::ReplyDraft` variants
- `MentionPoll` cron handler in the daemon dispatcher
- `ReplyDraft` handler that constructs `ReplyConfig` from registry + command payload
- `ReplyReviewDelivery` impl on `TelegramReviewDelivery` (parent-quoted message shape)
- Daemon TOML knobs (`[daemon.persona.<name>.mentions]`)
- Live test: post a tweet, get a colleague to reply, verify the Telegram review fires within 5 minutes and the picked reply lands

After P1.5c, the loop is closed: cron fires → mention shows up in Telegram → pick → reply posts.

---

## Appendix A — Worked example of the `reply_writer` user message

When the pipeline passes a mention to `reply_writer`, the user message looks like:

```
PARENT TWEET (yours, posted 2026-05-08T10:14:00Z):
> Implement two methods on a struct, heartbit-core wires up the entire tool pipeline.

THEIR REPLY (from @grumpy_dev, 1.2k followers, posted 2026-05-08T11:02:00Z):
> sure but how does it compare to rig-rs? feels like reinventing the wheel honestly

MENTIONER CONTEXT
- bio: "rust + ml. been around since 2015. rust-lang contributor."
- recent tweets: [3 tweets quoted, abridged to ~50 chars each]

VOICE GUIDELINES
[blended persona voice profile, same as run_pipeline]

EVANGELISM MODE — heartbit-core
[mode_addendum if persona supplies one]

Compose a single reply (≤280 chars) addressing their specific question (rig-rs comparison). Output the reply text only.
```

The reply_writer's expected output for this fixture:

```
rig-rs is a fine option. heartbit-core leans harder on the agent-runner side: builtin guardrails, retry classification, tool-output redaction. if you mostly want a Tool trait + dispatcher, rig is leaner. it's a tradeoff, not a winner.
```

Note: 280 chars max, addresses the specific question, takes a position without being defensive, no "great point" / "thanks" opener.

---

**Self-review notes (post-write 2026-05-09):**
- Spec covers the 8 design decisions made in brainstorming: separate pipeline (yes), 5 anti-spam rules, JSONL storage for V1, `since_id` in store, parent + mentioner context, Telegram message shape, 3-sub-phase decomposition, calibration-only V1.
- One ambiguity: whether `judge` runs at all when `candidates_per_reply == 1`. Decision: skip judge when N=1 (matches `run_pipeline` semantics). The plan task that builds `run_reply_pipeline` clarifies.
- One TBD: `twitter_get_tweet` is referenced in §6.2 but the existing tool family doesn't expose it directly — `TwitterSearchTool` could be used as a workaround, OR add a small `TwitterGetTweetTool` (single-tweet GET via `/2/tweets/:id`). The plan task scopes whether to add it as part of P1.5b or defer.
- The CLI subcommand `persona reply once` is ergonomically clean but adds a new operator-facing command. Plan task verifies clap structure matches the existing `persona run` pattern.
