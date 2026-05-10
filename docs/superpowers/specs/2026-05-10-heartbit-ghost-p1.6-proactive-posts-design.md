# heartbit-ghost P1.6 — Proactive posts design

## 1. Goal

Wire the existing review-mode tweet pipeline (`heartbit_ghost::review::run_review_pipeline`) into a daemon-driven loop that fires per persona on a configurable cadence, generates a topic via a sub-agent, drafts candidate threads, gates them through Telegram, and posts the chosen draft. Calibration mode only — every post gates on Telegram review (autonomous posting is out of scope).

This is the proactive-posting counterpart of P1.5 (which wired reactive replies). After P1.6 ships, a single daemon process can run both loops simultaneously per persona: replying to mentions every 5 min and posting on the operator's cadence.

## 2. Architecture overview

```
[[daemon.persona_posts]]            ─→ PersonaPostScheduler (one per enabled entry)
  persona = "heartbit-ghost:x"           tick (every interval, gated by active_hours)
  enabled = true                         │
  post_interval_seconds = 14400          ▼
  active_hours = "09:00-22:00"      DaemonCommand::PersonaPost { persona }
  candidates_per_draft = 3               │
  post_history_path = "..."              ▼ (DaemonCore consumer loop)
                                    handle_persona_post() — free function
                                    ├─ topic_generator sub-agent
                                    │    inputs: persona-declared (own tweets,
                                    │            mentions, repo state, ...)
                                    │    output: "topic" or "no_topic"
                                    ├─ if "no_topic": record skip, return
                                    └─ else: run_review_pipeline(topic)
                                            (existing P1.3c+P1.3d flow)
                                          → Telegram review (parent flow reused)
                                          → twitter_thread on user pick
                                          → record outcome in PostHistoryStore
```

Mirror exact de l'architecture P1.5 mentions:
- `PersonaPostScheduler` ↔ `MentionPollScheduler`
- `handle_persona_post` ↔ `handle_mention_poll` + `handle_reply_draft` (un seul handler car le flow est plus court)
- `PostHistoryStore` ↔ `MentionStore`
- `PersonaPostsConfig` ↔ `PersonaMentionsConfig`
- `MentionContext` étendu pour porter aussi les états posts

## 3. Files

### New files

- `crates/heartbit-ghost/src/agents/topic_generator.rs` — sub-agent recipe (system prompt + `topic_generator_recipe()`)
- `crates/heartbit-ghost/src/tools/user_tweets.rs` — new `TwitterUserTweetsTool` wrapping `GET /2/users/:id/tweets`
- `crates/heartbit-ghost/src/posts/mod.rs` — value types (`PostTopic`, `PostHistoryEntry`, `PostOutcome`)
- `crates/heartbit-ghost/src/posts/history.rs` — `PostHistoryStore` trait + `InMemoryPostHistoryStore` + `JsonlPostHistoryStore`
- `crates/heartbit-ghost/src/posts/topic_context.rs` — `TopicContextProvider` trait + `XGhostTopicContext` + `HeartbitRsXTopicContext`
- `crates/heartbit/src/daemon/persona_post.rs` — `PersonaPostScheduler` (interval + active hours)
- `crates/heartbit/src/daemon/persona_post_handler.rs` — `handle_persona_post` free function + `PersonaPostDeps`
- `docs/superpowers/specs/2026-05-10-heartbit-ghost-p1.6-proactive-posts-design.md` — this file
- `docs/superpowers/plans/2026-05-10-heartbit-ghost-p1.6-proactive-posts.md` — implementation plan (created by next phase)

### Modified files

- `crates/heartbit-ghost/src/agents/mod.rs` — re-export `topic_generator_recipe`
- `crates/heartbit-ghost/src/lib.rs` — `pub mod posts;`
- `crates/heartbit-ghost/src/heartbit_rs.rs` — declare repo-grounded topic generator inputs in expansion
- `crates/heartbit-ghost/src/lib.rs` (XGhostPersona) — declare X-grounded topic generator inputs
- `crates/heartbit-core/src/config/daemon.rs` — `PersonaPostsConfig` struct + `persona_posts: Vec<PersonaPostsConfig>` field on `DaemonConfig`
- `crates/heartbit-core/src/config/mod.rs` — re-export `PersonaPostsConfig`
- `crates/heartbit/src/daemon/types.rs` — `DaemonCommand::PersonaPost { persona }` variant + serde test
- `crates/heartbit/src/daemon/core.rs` — replace stub arm with real handler dispatch
- `crates/heartbit/src/daemon/mention_context.rs` — add `posts_entries: HashMap<String, PersonaPostEntry>` (per-persona post state) + `post_histories` collection
- `crates/heartbit/src/daemon/mod.rs` — `pub mod persona_post; pub mod persona_post_handler;` + re-exports
- `crates/heartbit/src/lib.rs` — re-export `PersonaPostScheduler`, `PersonaPostDeps`, `handle_persona_post`, `PersonaPostsConfig`
- `crates/heartbit-cli/src/daemon/mod.rs` — extend `build_mention_context` (or rename to `build_persona_context`) to construct `PostHistoryStore` per entry and spawn `PersonaPostScheduler`s alongside `MentionPollScheduler`s
- `crates/heartbit-cli/src/persona.rs` — add `PersonaCommand::Post { name }` (one-off CLI counterpart, analogous to `persona reply`)
- `crates/heartbit-cli/src/persona_review.rs` — add `post_config_from_env` helper if needed (CLI one-off only)

## 4. `topic_generator` sub-agent

### System prompt

```
You propose ONE specific topic worth a thread (or "no_topic" if nothing
fresh to say). Your inputs vary by persona — see the user message.

OUTPUT
Either a single line of plain text (the topic) — terse, ≤120 chars, no
preamble, no quotation marks — OR the literal string "no_topic" if:
- you've already covered every input
- nothing in the inputs warrants a thread
- the inputs are too thin to ground a substantive post

CONSTRAINTS
- The topic must be ground-able: the writer should be able to draft a
  thread without inventing facts. If you can't say what specific point
  to make, output "no_topic".
- Avoid duplicating recent posts. Recent posts are in your inputs.
- Avoid generic topics ("AI is changing everything"). Be specific
  ("calibrated abstention vs forced answers in tool-use loops").
- One topic only. The thread structure is the writer's job.
```

### Recipe shape

```rust
pub fn topic_generator_recipe() -> AgentConfig {
    AgentConfig {
        name: "topic_generator".to_string(),
        description: "Propose one specific thread topic (or 'no_topic') from \
                      pre-fetched static context.".to_string(),
        system_prompt: TOPIC_GENERATOR_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),   // pure text-in / text-out, no tools
        max_tokens: Some(512),
        reasoning_effort: Some("low".to_string()),
        ..super::stub_recipe("topic_generator")
    }
}
```

The generator has **no tools**. Context comes from the user message (pre-fetched by the handler).

## 5. Persona-declared topic context provider

Each persona declares HOW to assemble the topic-context string passed into the generator's user message. The agent itself is a singleton (one shared recipe, no per-persona override needed).

### `TopicContextProvider` trait (heartbit-ghost crate)

```rust
/// Builds the persona-specific block of context that goes into the
/// topic generator's user message. Called by `handle_persona_post`
/// before invoking the generator.
#[async_trait::async_trait]
pub trait TopicContextProvider: Send + Sync {
    /// Returns a multi-line plain-text block. Empty string is allowed
    /// (the generator falls back to the topic_brief in that case).
    async fn build_context(
        &self,
        deps: &TopicContextDeps<'_>,
    ) -> Result<String, anyhow::Error>;
}

pub struct TopicContextDeps<'a> {
    /// For X API calls (own_recent_tweets, mentions).
    pub credentials: Arc<dyn CredentialResolver>,
    /// Operator's X user_id (resolved at config load).
    pub operator_user_id: &'a str,
    /// Recent post history (for de-dup signal).
    pub recent_history: Vec<PostHistoryEntry>,
}
```

Or — using the same `Pin<Box<dyn Future>>` pattern as the rest of P1.5 (avoids `async-trait` dep). Implementation detail; the plan picks.

### `XGhostTopicContext` (default for heartbit-ghost:x)

Calls in sequence:
1. `GET /2/users/:id/tweets?max_results=10` — own recent posts (uses a NEW `TwitterUserTweetsTool` introduced in P1.6 — see §11)
2. `GET /2/users/:id/mentions?max_results=10` — recent mentions (reuses existing `TwitterMentionsTool`)

Renders:
```
RECENT POSTS (yours, last 10):
- [2026-05-08] "...text..."
- [2026-05-07] "...text..."
...

RECENT MENTIONS (last 10):
- @user1: "...text..."
- @user2: "...text..."
...

RECENT POST HISTORY (last 5 from store):
- [2026-05-09] topic: "calibrated abstention" → Posted
- [2026-05-08] topic: "tool failure cascades" → Posted
...
```

### `HeartbitRsXTopicContext` (default for heartbit-rs:x)

Calls `repo_inspect` (existing tool) for:
- Recent commit messages (last 10)
- Top-level CHANGELOG entries
- Recently-modified module names

Renders:
```
RECENT COMMITS:
- [sha1] feat(ghost): reply pipeline lifecycle wiring
- [sha2] feat(daemon): MentionPoll handler + cron
...

RECENT POST HISTORY (last 5):
...
```

### Default fallback (personas without provider)

If the persona's `PersonaExpansion::topic_context_provider` is `None`, the handler injects ONLY the recent post history + the optional `topic_brief` from config. Cheaper, less fresh — works without any X API or repo access. Useful for personas in early development.

### Wiring

`PersonaExpansion` gains an optional field:
```rust
pub topic_context_provider: Option<Arc<dyn TopicContextProvider>>,
```

`XGhostPersona::expand()` populates it with `Arc::new(XGhostTopicContext::new())`. `HeartbitRsXPersona::expand()` populates it with `Arc::new(HeartbitRsXTopicContext::new())`.

## 6. `PostHistoryStore` (heartbit-ghost crate)

Trait analogous to `MentionStore`. Tracks per-persona post history to avoid topic duplication.

```rust
pub trait PostHistoryStore: Send + Sync {
    /// Record that we just posted (or skipped) for a persona.
    fn record(
        &self,
        persona: &str,
        entry: PostHistoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + '_>>;

    /// Recent N entries for a persona, most recent first. Used by the
    /// topic generator's input — passed verbatim into the user message.
    fn recent<'a>(
        &'a self,
        persona: &'a str,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PostHistoryEntry>, StoreError>> + Send + 'a>>;

    /// Returns true if `topic` matches any recent entry's topic
    /// (case-insensitive equality on the canonical topic string).
    fn was_posted_recently<'a>(
        &'a self,
        persona: &'a str,
        topic: &'a str,
        within: chrono::Duration,
    ) -> Pin<Box<dyn Future<Output = Result<bool, StoreError>> + Send + 'a>>;
}

pub struct PostHistoryEntry {
    pub posted_at: DateTime<Utc>,
    pub topic: String,
    pub outcome: PostOutcome,  // Posted { tweet_id, url } | Skipped | TimedOut | NoTopic | GateRejected | PublishFailed
    pub tweet_id: Option<String>,
}
```

`InMemoryPostHistoryStore` (Vec, RwLock) and `JsonlPostHistoryStore` (append-only + replay at open) — same shape as P1.5's `MentionStore` impls.

Topic uniqueness: V1 uses case-insensitive string equality with a configurable lookback window (default 30 days). Semantic dedup is out of scope.

## 7. `PersonaPostsConfig` (heartbit-core)

```toml
[[daemon.persona_posts]]
persona = "heartbit-ghost:x"
enabled = true
post_interval_seconds = 14400               # 4 hours
active_hours = "09:00-22:00"                # optional; if absent, always active
candidates_per_draft = 3                    # 1..=10, like review mode
post_history_path = "~/.heartbit/ghost/posts/heartbit-ghost-x.jsonl"
post_history_lookback_days = 30             # how far back to check for topic dupes
topic_brief = "agent infrastructure, Rust, LLMs"  # optional fallback brief
```

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct PersonaPostsConfig {
    pub persona: String,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_post_interval_seconds")]
    pub post_interval_seconds: u64,
    #[serde(default)]
    pub active_hours: Option<String>,
    #[serde(default = "default_candidates_per_draft")]
    pub candidates_per_draft: usize,
    #[serde(default = "default_post_history_backend")]
    pub post_history_store: String,  // "in_memory" | "jsonl"
    #[serde(default)]
    pub post_history_path: Option<String>,
    #[serde(default = "default_post_history_lookback_days")]
    pub post_history_lookback_days: i64,
    #[serde(default)]
    pub topic_brief: Option<String>,
}
```

Defaults:
- `post_interval_seconds`: 14400 (4 hours) — conservative; X has no rate cap on posting per se, but high frequency degrades engagement
- `candidates_per_draft`: 3 — same as `persona run --review`
- `post_history_store`: `"in_memory"` — same convention as mentions
- `post_history_lookback_days`: 30
- `active_hours` parsing reuses `ActiveHoursConfig::parse_start/parse_end` (already in the daemon config module)

## 8. Daemon command + handler

### Command variant

```rust
pub enum DaemonCommand {
    // ... existing variants
    /// Cron-driven: generate one proactive post for `persona`. Fires
    /// from `PersonaPostScheduler` on the configured cadence (gated by
    /// active_hours).
    PersonaPost {
        persona: String,
    },
}
```

Single-field variant (per-persona scoping) — no need for `user_id` since posting acts on the OAuth1-authenticated account (no per-tweet operator selection).

### Scheduler

```rust
pub struct PersonaPostScheduler {
    persona: String,
    interval: Duration,
    active_hours: Option<ActiveHoursConfig>,
    producer: Arc<dyn CommandProducer>,
    commands_topic: String,
}
```

Run loop reuses the `is_within_active_hours` helper. If outside active hours, sleep one tick and re-check (no command fired). If inside, fire `DaemonCommand::PersonaPost { persona }` to the commands topic via the producer.

### Handler

```rust
pub struct PersonaPostDeps<'a> {
    pub persona_name: &'a str,
    pub registry: &'a PersonaRegistry,
    pub history: &'a dyn PostHistoryStore,
    pub history_lookback: chrono::Duration,
    pub topic_brief: Option<&'a str>,
    /// Operator's X user_id, needed by XGhost-style topic context providers
    /// for own-tweets / mentions fetches.
    pub operator_user_id: &'a str,
    pub provider: Arc<BoxedProvider>,
    pub delivery: Arc<dyn ReviewDelivery>,
    /// `twitter_thread` tool — used by run_review_pipeline to post the chosen draft.
    pub twitter_tool: Arc<dyn Tool>,
    pub credentials: Arc<dyn CredentialResolver>,
    pub candidates_per_draft: usize,
    pub corpora_root: &'a Path,
    pub profiles_root: &'a Path,
}

pub async fn handle_persona_post<'a>(
    deps: PersonaPostDeps<'a>,
) -> Result<PostOutcome, anyhow::Error>;
```

Body (sketch):

1. `persona = deps.registry.get(deps.persona_name)?` then `.expand()` to get `topic_context_provider`, `researcher_override`, `mode_addendum`.
2. Build the topic-generator's input context:
   - If `expansion.topic_context_provider` is `Some(provider)`: call `provider.build_context(&deps)` for the persona-specific block (own tweets / mentions / repo state).
   - Else: empty string.
   - Always append the recent post history (last 5 entries, abridged) — independent of provider.
   - Append `topic_brief` if set in config.
3. Build `topic_generator` runner from `topic_generator_recipe()` with NO tools.
4. Run topic generator with the assembled context as user message. If output (trimmed, lowercased) starts with "no_topic" → `record(NoTopic)`, return `PostOutcome::NoTopic`.
5. Else, check `was_posted_recently(topic, history_lookback)`. If yes → `record(SkippedDuplicate)`, return.
6. Else, build `ReviewConfig` from the topic + persona expansion (`mode_addendum`, `researcher_override`) and call `run_review_pipeline`.
7. Map the `ReviewOutcome` to a `PostOutcome` and `record()` it.

Recording happens regardless of outcome (Posted / Skipped / TimedOut / GateRejected / PublishFailed / NoTopic / SkippedDuplicate) so the history accurately reflects all decisions.

## 9. Lifecycle wiring

### `MentionContext` extension

Rename `MentionContext` → `PersonaContext` (or keep the name and treat it as the umbrella). Add fields:

```rust
pub struct MentionContext {
    // ... existing fields (registry, provider, twitter_mentions, twitter_reply,
    //                     credentials, corpora_root, profiles_root, delivery)
    /// Per-persona mention state.
    pub mentions_entries: HashMap<String, PersonaMentionEntry>,
    /// Per-persona post state.
    pub posts_entries: HashMap<String, PersonaPostEntry>,
}

pub struct PersonaPostEntry {
    pub history: Arc<dyn PostHistoryStore>,
    pub interval: Duration,
    pub active_hours: Option<ActiveHoursConfig>,
    pub candidates_per_draft: usize,
    pub history_lookback: chrono::Duration,
    pub topic_brief: Option<String>,
}
```

Reusing `MentionContext`'s shared resources (registry, provider, delivery, credentials, profile roots, twitter tools) — avoids duplication. The `delivery` is shared: same Telegram chat receives both reply reviews and post reviews; the user sees them interleaved.

### `DaemonCore::run` startup spawn

After spawning `MentionPollScheduler` instances, also spawn `PersonaPostScheduler` instances per `posts_entries`:

```rust
for (persona, entry) in ctx.posts_entries.iter() {
    let scheduler = PersonaPostScheduler::new(
        persona.clone(),
        entry.interval,
        entry.active_hours.clone(),
        producer.clone(),
        &commands_topic,
    );
    tokio::spawn(scheduler.run(self.cancel.clone()));
}
```

### `DaemonCommand::PersonaPost` arm

```rust
DaemonCommand::PersonaPost { persona } => {
    let Some(ctx) = self.mention_context.clone() else { /* warn + skip */ };
    let Some(entry) = ctx.posts_entries.get(&persona) else { /* warn + skip */ };
    let history = entry.history.clone();
    let history_lookback = entry.history_lookback;
    let topic_brief = entry.topic_brief.clone();
    let candidates_per_draft = entry.candidates_per_draft;
    let registry = ctx.registry.clone();
    let provider = ctx.provider.clone();
    let delivery = ctx.delivery.clone();
    let twitter_thread = ctx.twitter_thread.clone();  // NEW: needed alongside twitter_reply
    let credentials = ctx.credentials.clone();
    let corpora_root = ctx.corpora_root.clone();
    let profiles_root = ctx.profiles_root.clone();
    let persona_owned = persona.clone();
    tokio::spawn(async move {
        let deps = PersonaPostDeps { /* ... */ };
        if let Err(e) = handle_persona_post(deps).await {
            tracing::error!(error = %e, "persona post handler failed");
        }
    });
}
```

`twitter_thread` is currently absent from `MentionContext` (P1.5 only needed `twitter_reply` + `twitter_mentions`). P1.6 adds it.

## 10. CLI

### `persona post <name>` — on-demand counterpart

Mirror of `persona reply`. Triggers `handle_persona_post` once with the configured deps. Useful for testing the topic generator + pipeline without waiting for the cron tick.

```rust
PersonaCommand::Post {
    name: String,
    /// Override the configured candidates_per_draft.
    #[arg(long)]
    candidates: Option<usize>,
    /// Override the topic. When set, skips the topic_generator step.
    #[arg(long)]
    topic: Option<String>,
}
```

### `persona posts list <name>` — recent post history (operator polish)

Mirror of `persona mentions`. Reads `PostHistoryStore` and prints the last N entries.

## 11. Test plan

Unit tests (TDD throughout):

- `topic_generator_recipe_has_expected_shape` (same shape as reply_writer test)
- `topic_generator_prompt_mandates_no_topic_escape`
- `TwitterUserTweetsTool` happy path + 401 + rate limit (mirror `TwitterMentionsTool`'s test layout)
- `XGhostTopicContext::build_context` — assembles own-tweets + mentions + history block from mocked tools
- `HeartbitRsXTopicContext::build_context` — assembles repo-state block from mocked `RepoInspectTool`
- `PostHistoryStore` impls — record + recent + was_posted_recently round-trip; jsonl reload; lookback window filtering
- `PersonaPostScheduler` — fires `PersonaPost` on tick; respects `active_hours`; cancels cleanly
- `PersonaPostsConfig` deserialize defaults
- `DaemonCommand::PersonaPost` serde round-trip

Integration tests (mirror P1.5 patterns):

- `handle_persona_post_happy_path_runs_pipeline_and_records` — topic returned, pipeline returns `Posted`, history records `Posted { tweet_id, ... }`
- `handle_persona_post_no_topic_short_circuits` — generator returns "no_topic", pipeline NOT called, history records `NoTopic`
- `handle_persona_post_duplicate_topic_skips_pipeline` — generator returns a topic seen 5 days ago (within lookback), pipeline NOT called, history records `SkippedDuplicate`
- `handle_persona_post_telegram_skip_records_skipped` — pipeline runs, user picks Skip, history records `Skipped`
- `handle_persona_post_pipeline_failure_propagates_err` — pipeline errors, history NOT recorded (preserves replay safety)

Acceptance:

- Quality gate (`cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace --features daemon`)
- Live test:
  1. Configure `[[daemon.persona_posts]]` with `post_interval_seconds = 60` (1 min for the test) and `active_hours` covering now
  2. Start `heartbit daemon --config <toml>`
  3. Within 60s: Telegram receives the post review with 2-3 candidates + `[1] [2] [3] [Skip]` keyboard
  4. Pick a draft → reply posts on X
  5. Verify `~/.heartbit/ghost/posts/heartbit-ghost-x.jsonl` contains a `Posted` entry with the tweet_id
  6. Wait another 60s: second tick fires; topic generator either picks a new topic or returns "no_topic" (we expect "no_topic" if context is unchanged — the prompt instructs it to avoid duplicates)

## 12. Out of scope (explicit)

- Auto-post (no Telegram review) — calibration mode only
- Semantic topic dedup — V1 uses case-insensitive string equality
- Cron-style scheduling — interval + active_hours only (matches P1.5 mention poll cadence)
- Multi-persona orchestration ("don't post if persona X just posted") — each persona independent
- Idle backoff on `no_topic` — predictable cadence preferred; the topic generator is cheap so a wasted tick is acceptable
- Topic re-feedback ("the last 3 mentions all asked about X, post about X") — generator decides; no closed loop with the mention queue
- Multi-tweet thread structure — the writer (existing pipeline) decides single tweet vs thread
- Image attachments on proactive posts — out of scope for V1; the pipeline already supports them via `image_generator` for future iterations

## 13. Risks

| Risk | Mitigation |
|------|-----------|
| Topic generator returns same topic repeatedly | History injected into generator's user message + duplicate check before pipeline |
| Telegram chat floods if user steps away | `active_hours` gates outside hours; user can also Skip every review (recorded as `Skipped`, lookback prevents immediate retry) |
| Daemon spam if `post_interval_seconds = 0` or very small | Validate at config load: `post_interval_seconds < 60` rejected with a clear error |
| Provider cost from "no_topic" ticks | Topic generator is cheap (max_tokens=512, reasoning=low); ticks that return "no_topic" cost ~$0.001 at current rates |
| Concurrent mention reply + proactive post on same persona | Both use the same shared `delivery` (single Telegram dispatcher); reviews interleave naturally; no cross-coupling. The persona's voice profile is read-only at runtime so concurrent pipelines are safe. |
| `PostHistoryStore` write race across concurrent posts | `record()` is async + lock-protected; one persona never has two concurrent post handler invocations because the scheduler fires sequentially |
| Tilde expansion in `post_history_path` | Reuse the existing `expand_tilde` helper from the CLI's daemon module (added in P1.5 task 13) |

## 14. Sub-phases

- **P1.6a** — `TwitterUserTweetsTool` + `topic_generator` recipe + `TopicContextProvider` trait + `XGhostTopicContext` + `HeartbitRsXTopicContext` + `topic_context_provider` field on `PersonaExpansion` (Tasks 1-5)
- **P1.6b** — `PostHistoryStore` (trait + 2 impls) + `PersonaPostsConfig` + `DaemonCommand::PersonaPost` (Tasks 6-8)
- **P1.6c** — `PersonaPostScheduler` + `handle_persona_post` + lifecycle wiring + CLI `persona post` and `persona posts list` (Tasks 9-12)
- **P1.6d** — Acceptance: quality gate + live test + docs (Task 13)

Each sub-phase ships with passing tests at the end. P1.6a can land standalone (no behavior change yet — just plumbing + a new tool). P1.6b is internal-only. P1.6c is the user-visible step. P1.6d is gating only.

## 15. Open questions

None at design freeze. All judgment calls resolved during brainstorming:
- Topic source: generator agent ✅
- Generator input: persona-declared ✅
- Cadence: interval + active hours ✅
- Review mode: Telegram calibration (always) ✅
- "no_topic" handling: skip tick, record, no auto-retry ✅

Future iterations (out of P1.6 scope) could add: semantic dedup, multi-persona coordination, autonomous-mode posting (gated by autonomy phase from P1.4 work).
