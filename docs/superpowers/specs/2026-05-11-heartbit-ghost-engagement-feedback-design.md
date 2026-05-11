# Heartbit-Ghost — Engagement Feedback Loop (Design)

**Date**: 2026-05-11
**Phase**: P2.0 (post-P1.7)
**Status**: Design (approved); ready for plan.

## Goal

Make the proactive-post `writer` agent learn from your X audience by feeding it your top-engaged recent posts as few-shot exemplars. Today every generation is cold — the writer has no memory of what your specific followers actually engage with. After this feature, the writer's prompt dynamically includes the 5 highest-engaged posts from the last 30 days as voice + structure examples.

This converts the ghostwriter from a stylistically-consistent drafter into one that converges on the principal's *audience*. It is the single biggest expected quality lift available; everything else in the existing pipeline (scam judge, enrichment cache, jitter) is plumbing-grade rather than capability-grade.

## Motivation

Existing writer behavior:
1. Static `system_prompt` defined in `agents/writer.rs`
2. No knowledge of past post outcomes beyond "this topic was posted recently" (duplicate suppression)
3. Generates each candidate cold, with only persona-level voice profile (`crates/heartbit-ghost/src/voice/`)

Observed gap (post-P1.6 smoke testing): the writer drafts technically-correct threads in the persona's voice, but the operator must repeatedly skip drafts that miss the audience's actual interests. Each skip is a strong negative signal that the system discards. Each *post* that gets above-median engagement is a strong positive signal that the system also discards. Both signals should compound.

## Non-Goals (deliberate, deferrable)

- **Engagement judge**: a separate LLM that scores drafts and rejects low-predicted ones. Adds one LLM call per candidate + drift risk if the judge is wrong. The few-shot path provides positive guidance; we don't need a negative filter in V1.
- **Topic-generator biasing**: weighting topic proposals toward themes with historical engagement. Orthogonal — can be added if writer-only feedback doesn't move the needle.
- **Reply-writer learning**: replies have noisy attribution (engagement depends heavily on the parent tweet's audience, not pure voice). V1 is proactive-only.
- **Engagement-trend memory**: storing snapshot history for long-term analysis. V1 keeps just the latest snapshot per tweet.
- **Quote-tweet integration**: a separate content vector; tracked as a follow-up feature.

## Architecture

Four modular layers, each independently testable:

```
┌─────────────────────────────────────────────────────────┐
│  EngagementCollector    (scheduler, runs every ~6h)     │
│      │                                                  │
│      ▼ batch GET /2/tweets?ids=...&tweet.fields=...     │
│  EngagementStore        (JSONL append-only)             │
│      │                                                  │
│      ▼ latest snapshot per tweet_id                     │
│  TopPostsProvider       (composite score, top N)        │
│      │                                                  │
│      ▼ Vec<TopPost> with text                           │
│  persona_post_handler   (prepends exemplars to writer)  │
└─────────────────────────────────────────────────────────┘
```

The four layers connect by ID. None of them know about each other's internals beyond a small trait surface.

## Component Specifications

### 1. `EngagementSnapshot` + `EngagementStore`

**Location**: `crates/heartbit-ghost/src/posts/engagement.rs`

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EngagementSnapshot {
    pub tweet_id: String,
    pub captured_at: DateTime<Utc>,
    pub likes: u64,
    pub replies: u64,
    pub retweets: u64,
    pub quotes: u64,
    pub bookmarks: u64,
    /// X public_metrics now exposes impression_count, but treat as
    /// Optional — older snapshots written before impressions were
    /// available, or tweets where the API omits the field, should
    /// still parse.
    pub impressions: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum EngagementStoreError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse: {0}")]
    Parse(String),
}

pub trait EngagementStore: Send + Sync {
    /// Append a snapshot. Implementations are append-only on disk.
    fn record<'a>(
        &'a self,
        snap: EngagementSnapshot,
    ) -> Pin<Box<dyn Future<Output = Result<(), EngagementStoreError>> + Send + 'a>>;

    /// Return the latest snapshot per tweet_id. Used by TopPostsProvider
    /// to rank without scanning the full append log.
    fn latest_per_tweet<'a>(
        &'a self,
    ) -> Pin<Box<dyn Future<Output = Result<HashMap<String, EngagementSnapshot>, EngagementStoreError>> + Send + 'a>>;
}
```

Two implementations (parallel to `MentionStore`):
- `InMemoryEngagementStore` — for tests; replayable in-process
- `JsonlEngagementStore::open(path)` — for production; replays JSONL on construction, appends one event per `record`

JSONL event format (mirrors `StoreEvent` in `reply/storage.rs`):
```json
{"event":"snapshot","tweet_id":"123","captured_at":"...","likes":42, ...}
```

**Storage path**: `.heartbit/engagement/heartbit-ghost-x.jsonl`

### 2. `engagement::collector` — periodic refresh helper

**Location**: `crates/heartbit-ghost/src/posts/engagement.rs` (same module, free function)

```rust
/// Refresh engagement metrics for every Posted entry in the last
/// `max_age_days` that's at least `min_age_hours` old. Skips entries
/// where `tweet_id.is_none()`. Batches by 100 ids per X API call.
pub async fn refresh_engagement(
    client: &XClient,
    history: &dyn PostHistoryStore,
    store: &dyn EngagementStore,
    persona: &str,
    now: DateTime<Utc>,
    max_age_days: i64,
    min_age_hours: i64,
) -> Result<RefreshOutcome, RefreshError>;

pub struct RefreshOutcome {
    pub queried: usize,
    pub refreshed: usize,
    pub skipped_too_young: usize,
    pub failed: usize,
}
```

Logic:
1. Pull `Posted` entries from history within `[now - max_age_days, now - min_age_hours]`
2. Collect their `tweet_id`s
3. Batch by 100 (X API limit) → call `GET /2/tweets?ids=...&tweet.fields=public_metrics`
4. For each tweet in response, build + `store.record(snapshot)`
5. Tweets missing in the response (e.g. deleted) are counted in `failed`; not retried

`min_age_hours=24` default — engagement matures fast on X, but a freshly posted tweet shouldn't enter the top-N pool while still rising.

### 3. `EngagementCollectorScheduler`

**Location**: `crates/heartbit/src/daemon/engagement_collector.rs`

Mirrors `PersonaPostScheduler` and `MentionPollScheduler`:
- Configurable interval (`engagement_refresh_seconds`, default 21600 = 6h)
- Same `±jitter_pct` randomization (default 25%, reusing the existing config field shape)
- No `active_hours` gate — engagement collection is cheap and can run at any hour
- On each tick: calls `refresh_engagement`, logs `RefreshOutcome` at INFO

Dispatch via Kafka command (same pattern as PersonaPost): `DaemonCommand::EngagementRefresh { persona }`. The daemon consumer side calls a new free-function handler `handle_engagement_refresh`.

### 4. `TopPostsProvider`

**Location**: `crates/heartbit-ghost/src/posts/engagement.rs`

```rust
#[derive(Debug, Clone)]
pub struct TopPost {
    pub tweet_id: String,
    pub text: String,
    pub posted_at: DateTime<Utc>,
    pub engagement_score: f64,
}

pub trait TopPostsProvider: Send + Sync {
    async fn top_n(&self, n: usize) -> Result<Vec<TopPost>, EngagementStoreError>;
}

pub struct JoinedTopPostsProvider {
    history: Arc<dyn PostHistoryStore>,
    engagement: Arc<dyn EngagementStore>,
}

impl JoinedTopPostsProvider {
    pub fn new(history: Arc<dyn PostHistoryStore>, engagement: Arc<dyn EngagementStore>) -> Self;
}
```

Composite score (replies and retweets reflect deeper engagement than passive likes):
```
score = likes
      + 3.0 * replies
      + 2.0 * retweets
      + 2.0 * quotes
      + impressions.map(|i| 0.0001 * (i as f64)).unwrap_or(0.0)
```
Returns top N posts by score, descending. Ties broken by `posted_at` descending (prefer recency).

`top_n` joins `EngagementStore::latest_per_tweet` against `PostHistoryStore::recent(persona, large_n)` (e.g. 1000) to get the `text`. Build a `HashMap<tweet_id, text>` from the history entries with a present `tweet_id` and a present `text`, then iterate the engagement snapshots in score order and drop any that don't have a matching text. No new method on `PostHistoryStore` is required. If a history entry's `text` is missing (older entries from before this feature), the tweet is skipped — we never produce a `TopPost` without text.

### 5. Extend `PostHistoryEntry` with `text`

**Location**: `crates/heartbit-ghost/src/posts/mod.rs`

```rust
pub struct PostHistoryEntry {
    pub posted_at: DateTime<Utc>,
    pub topic: String,
    pub outcome: PostOutcome,
    pub tweet_id: Option<String>,
    /// First tweet of the thread (or single tweet text). Captured at post
    /// time so `TopPostsProvider` doesn't need to round-trip the X API
    /// just to render exemplars. `#[serde(default)]` for backward
    /// compatibility with entries written pre-P2.0.
    #[serde(default)]
    pub text: Option<String>,
}
```

Recorded by `persona_post_handler` when the pipeline returns `Posted` — the first candidate text is already in-memory at that point.

### 6. Writer prompt injection

**Location**: `crates/heartbit/src/daemon/persona_post_handler.rs`

Just before invoking the writer agent (currently inside `run_review_pipeline` / the multi-agent recipe), build a dynamic exemplar block:

```rust
let exemplars = top_posts_provider.top_n(top_n).await.unwrap_or_default();
let exemplar_block = if exemplars.len() >= 3 {
    let mut s = String::from(
        "EXEMPLARS — your highest-engaged posts from the last 30 days.\n\
        Study the voice, structure, and angle. Do NOT copy literally.\n\n"
    );
    for p in &exemplars {
        let age = (now - p.posted_at).num_days();
        s.push_str(&format!(
            "[{} days ago, engagement score {:.0}]\n{}\n\n",
            age, p.engagement_score, p.text
        ));
    }
    s.push_str("---\n");
    s
} else {
    String::new() // cold start: fewer than 3 exemplars → no injection
};

// Prepend exemplar_block to writer's user_message.
```

Critical: exemplars are added to the **user message**, NOT the system prompt. Two reasons:
1. The recipe's `system_prompt` is a `const`; keeping it static makes the recipe testable and serializable.
2. Anthropic's prompt-caching boundary is the system prompt — varying system prompt every call would defeat caching. User-message variance is expected and free.

Cold-start behavior: when there are < 3 exemplars (fresh deployment, first few posts), the block is empty and the writer falls back to existing behavior. No regression.

### 7. Config schema (per-persona)

Extend `PersonaPostsConfig` in `heartbit-core/src/config/daemon.rs`:

```rust
pub struct PersonaPostsConfig {
    // ... existing fields ...

    /// Engagement-collector tick interval (seconds). Default 21600 = 6h.
    #[serde(default = "default_engagement_refresh_seconds")]
    pub engagement_refresh_seconds: u64,

    /// How many top-engaged posts to inject as exemplars. Default 5.
    /// Set to 0 to disable the feature entirely.
    #[serde(default = "default_engagement_top_n")]
    pub engagement_top_n: usize,

    /// Ignore tweets older than this many days. Default 30.
    #[serde(default = "default_engagement_max_age_days")]
    pub engagement_max_age_days: i64,

    /// Ignore tweets younger than this many hours (engagement
    /// hasn't matured). Default 24.
    #[serde(default = "default_engagement_min_age_hours")]
    pub engagement_min_age_hours: i64,
}
```

All four have sensible defaults; existing configs continue to work unchanged.

## Data flow

```
                  daemon startup
                       │
                       ▼
        ┌──────────────────────────────┐
        │ EngagementCollectorScheduler │  every 6h ±25%
        └──────────────────────────────┘
                       │ dispatch
                       ▼ DaemonCommand::EngagementRefresh
            handle_engagement_refresh
                       │
                       ▼
           refresh_engagement(persona)
                       │
                       ▼ batch fetch
            GET /2/tweets?ids=...
                       │
                       ▼ for each result
           EngagementStore::record(snap)
                       │
                       ▼ (JSONL append)
   .heartbit/engagement/heartbit-ghost-x.jsonl


                  proactive post tick
                       │
                       ▼
              persona_post_handler
                       │
                       ▼ before writer invocation
        top_posts_provider.top_n(5)
                       │ joins:
                       ▼
   ┌────────────────────────────┐
   │ JsonlEngagementStore       │  latest snapshots
   │ JsonlPostHistoryStore      │  tweet text + posted_at
   └────────────────────────────┘
                       │
                       ▼ composite score, sorted
        Vec<TopPost> { text, score, age }
                       │
                       ▼ prepend to user_message
              writer agent invocation
```

## Error handling / edge cases

| Scenario | Behavior |
|---|---|
| X API returns 429 in refresh_engagement | Back off; do not crash. Counter increments `failed`. Next 6h tick retries. |
| X API returns 404 for a tweet_id (deleted tweet) | Skip silently. Counter increments `failed`. No retry — tweet is gone. |
| `EngagementStore::record` fails (disk full) | Warn-log; don't propagate. Next refresh will retry the snapshot. |
| `latest_per_tweet` returns empty (fresh deployment) | `top_n` returns empty vec → writer falls back to no-exemplar path. |
| `PostHistoryEntry.text` is `None` (pre-feature entries) | Skip that entry in `top_n` join — we never inject a TopPost without text. |
| `engagement_top_n = 0` in config | Provider returns empty vec on every call → exemplar block is empty → writer unaffected. Effectively disables the feature. |
| Daemon crash mid-refresh | JSONL is append-only and crash-safe; some tweets may have been recorded, some not. Next refresh picks up where it left off (idempotent — same tweet_id is fine to re-snapshot). |

## Testing strategy

Each layer is independently testable:

1. **`EngagementSnapshot` + `JsonlEngagementStore`**: 5 tests (round-trip, replay-across-reload, `latest_per_tweet` returns most recent, missing-file handles cleanly, backward-compat parsing of snapshots without `impressions`).

2. **`refresh_engagement`**: 5 tests using wiremock for X API (happy path with 3 tweets, skip-too-young, partial-failure when one tweet 404s, batch boundary at 100 ids, empty history → no API call).

3. **`JoinedTopPostsProvider`**: 4 tests (top-N order by composite score, ties broken by recency, missing-text entries excluded, impressions-optional weighting).

4. **`EngagementCollectorScheduler`** + Kafka dispatch: 3 tests (fires at jittered interval, dispatches correct `DaemonCommand`, cancels cleanly).

5. **`handle_engagement_refresh`**: 2 tests (calls refresh_engagement with correct args, unknown persona returns warn).

6. **End-to-end writer injection**: 1 integration test in `persona_post_handler` — pre-populate engagement store + history, run handler, assert the writer agent received exemplar block (verifiable via the test MockProvider's `captured_request`).

Target: ~20 new tests; all backends + wiring + integration covered.

## Estimated scope

5 tasks → 1 subagent-driven session:

1. **Storage layer** — `EngagementSnapshot`, `EngagementStore` trait + `InMemory` + `Jsonl` impls + tests
2. **Refresh helper** — `refresh_engagement` free function with wiremock tests
3. **Scheduler + handler** — `EngagementCollectorScheduler`, `DaemonCommand::EngagementRefresh`, `handle_engagement_refresh`, core.rs wiring
4. **Extend `PostHistoryEntry` with `text`** + record at post time
5. **`TopPostsProvider` + writer injection** — composite score, integration into `persona_post_handler`, config plumbing through CLI
