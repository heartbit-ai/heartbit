# heartbit-ghost P1.3d — Telegram review delivery + publisher (twitter_thread direct) + Phase 0 default

**Status:** approved 2026-05-08
**Predecessor:** P1.3c (multi-candidate + judge + image_generator) merged to `main` at `fdd7bac`
**Branch:** `feat/heartbit-ghost-p1.3d` (created off `main`)
**Successor:** P1.3e — Persistent pick storage + reflection memory + few-shot exemplar retrieval

---

## 1. Goal

Add a "review mode" entry point to heartbit-ghost where N candidate drafts are sent to the user via Telegram, the user picks one (or skips, or times out), and on a positive pick the chosen draft is posted to X via `twitter_thread` directly (no publisher LLM). The CLI gains a `--review` flag that selects this path.

This delivers the umbrella spec §3 / §6.2 / §6.4 (Phase 0 calibration) end-to-end: real LLM-generated candidates → real user review on Telegram → real tweet posted, with no auto-publish path. Phase 0 is the **only** mode in P1.3d (no auto-publish exists yet); auto-publish (Phase 1+) lands in P1.4.

## 2. Architecture

```
heartbit persona run heartbit-ghost:x --once "<topic>" --review
                              │
                              ▼
            run_review_pipeline(cfg)
                              │
              ┌───────────────┴────────────────────────┐
              │ shared with run_pipeline (P1.3c):      │
              │ snapshot load → research → 3× parallel │
              │ writer→critic→fact_check → dedup       │
              └───────────────┬────────────────────────┘
                              │
                              │ (judge / image_generator / publish_gate / println skipped)
                              │
                              ▼
              build_review_message → ReviewMessage
                              │
                              ▼
              ReviewDelivery::deliver_and_await(message)
                  │                    (real impl: TelegramReviewDelivery; test impl: MockReviewDelivery)
                  ▼
              (DeliveryOutcome, DeliveryReceipt)
                  │
       ┌──────────┼──────────────┐
       ▼          ▼              ▼
     Pick(i)    Skip          TimedOut
       │          │              │
       ▼          │              │
     publish_gate(candidates[i].draft)
       │          │              │
       │ Err──────┼──────────────┼──┐
       │          │              │  │
       ▼          │              │  │
     parse_thread_tweets(draft)  │  │
       │          │              │  │
       ▼          │              │  │
     TwitterThreadTool::execute(ctx, tweets[]) ──┐
       │          │              │  │            │
       │ Err──────┼──────────────┼──┼────────────┤
       │          │              │  │            │
       ▼          ▼              ▼  ▼            ▼
     ReportableOutcome::Posted | Skipped | TimedOut | GateRejected | PublishFailed
                              │
                              ▼
              ReviewDelivery::report(receipt, ReportableOutcome)
                  (Telegram impl edits the original message in place)
                              │
                              ▼
                  return ReviewOutput { outcome: ReviewOutcome, ... }
```

**Shared with `run_pipeline` (P1.3c):** snapshot load, agent construction (researcher/writer/style_critic/fact_check only — no judge/image_generator), parallel candidate generation, dedup with bounded retry. After dedup the path forks: direct mode runs judge/image/gate/println; review mode goes to delivery.

**Phase 0 default:** there is no auto-publish path in P1.3d. `--review` is the **only** way to actually post a tweet. The flag default is `false` (preserves P1.3c's stdout-only behavior); `--review` opts into Telegram + posting.

## 3. Public API

### 3.1 New module `heartbit_ghost::review`

```rust
// crates/heartbit-ghost/src/review/mod.rs
pub use crate::pipeline::CandidateRecord;

/// Configuration for one review-mode pipeline run.
#[derive(Clone)]
pub struct ReviewConfig<'a> {
    /// Persona instance name (used to load StyleProfile snapshot).
    pub persona_name: &'a str,
    /// Topic / prompt for this run.
    pub topic: &'a str,
    /// LLM provider (shared across sub-agents).
    pub provider: Arc<BoxedProvider>,
    /// Corpora root (currently unused; reserved for P1.3e).
    pub corpora_root: &'a Path,
    /// Profiles root (passed to SnapshotStore::open).
    pub profiles_root: &'a Path,
    /// Optional progress callback.
    pub on_progress: Option<ProgressCallback>,
    /// Number of distinct candidate drafts to generate. Same semantics as
    /// PipelineConfig.candidates_per_draft (validated 1..=10).
    pub candidates_per_draft: usize,
    /// Telegram-or-mock delivery layer. Production wires
    /// `TelegramReviewDelivery`; tests wire `MockReviewDelivery`.
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
    /// Accumulated token usage across sub-agent calls (judge / image_generator
    /// excluded since they're skipped in review mode).
    pub usage_summary: TokenUsage,
    /// What happened.
    pub outcome: ReviewOutcome,
}

/// Outcome of the review interaction.
#[derive(Debug, Clone)]
pub enum ReviewOutcome {
    /// User picked candidate `chosen_index` and the post was published.
    Posted {
        /// Index into `candidates`.
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
    /// User picked candidate `chosen_index` but `publish_gate` rejected the draft.
    GateRejected {
        chosen_index: usize,
        reason: String,
    },
    /// User picked candidate `chosen_index` but the X API call failed.
    PublishFailed {
        chosen_index: usize,
        reason: String,
    },
}

/// Errors raised by [`run_review_pipeline`].
#[derive(Debug, Error)]
pub enum ReviewError {
    /// Candidate generation failed (delegates to PipelineError variants).
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    /// Telegram (or mock) delivery failed.
    #[error("delivery: {0}")]
    Delivery(#[from] ReviewDeliveryError),
    /// `PipelineConfig` validation at run start.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

// NOTE: `ReviewDelivery::report()` failures are intentionally NOT a
// `ReviewError` variant. They're non-fatal (the post may have succeeded;
// only the after-the-fact Telegram message edit failed) and are logged
// via the `on_progress` callback. `run_review_pipeline` returns
// `Ok(ReviewOutput { outcome, ... })` even when `report()` fails — the
// outcome is still authoritative.

/// Top-level entry point for review-mode pipeline.
pub async fn run_review_pipeline(cfg: ReviewConfig<'_>) -> Result<ReviewOutput, ReviewError>;
```

### 3.2 `ReviewDelivery` trait

```rust
// crates/heartbit-ghost/src/review/delivery.rs
use async_trait::async_trait;
use serde_json::Value;

/// Abstracts the user-interaction layer (Telegram, mock, future channels).
/// `heartbit-ghost` doesn't depend on `teloxide`; production impl lives in
/// `heartbit-cli` (which already has the dep via `heartbit-telegram`).
#[async_trait]
pub trait ReviewDelivery: Send + Sync {
    /// Send candidates to the user, wait for their pick (or timeout).
    /// Returns the outcome + an opaque receipt the impl uses to correlate
    /// a later `report()` call (Telegram = chat_id+message_id).
    async fn deliver_and_await(
        &self,
        message: &ReviewMessage,
    ) -> Result<DeliveredReview, ReviewDeliveryError>;

    /// Update the prior delivery with the final result (e.g., Telegram
    /// edits the message in place to remove the keyboard and show
    /// "Posted: <url>" / "Skipped" / etc.). Implementations may noop if
    /// the medium doesn't support editing.
    async fn report(
        &self,
        receipt: DeliveryReceipt,
        outcome: ReportableOutcome,
    ) -> Result<(), ReviewDeliveryError>;
}

/// Input to delivery — the rendered message + correlation id.
#[derive(Debug, Clone)]
pub struct ReviewMessage {
    /// Persona instance name (rendered in the header).
    pub persona_name: String,
    /// Topic / shortname (rendered in the header).
    pub topic: String,
    /// Pre-rendered candidate drafts (Telegram-friendly, max ~3000 chars total).
    pub candidates: Vec<String>,
    /// UUID for keyboard callback correlation.
    pub interaction_id: Uuid,
}

#[derive(Debug)]
pub struct DeliveredReview {
    pub outcome: DeliveryOutcome,
    pub receipt: DeliveryReceipt,
}

#[derive(Debug, Clone)]
pub struct DeliveryReceipt {
    /// Impl-defined opaque payload. Telegram puts
    /// `{"chat_id": <i64>, "message_id": <i32>}`. Mock can put `null`.
    pub data: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeliveryOutcome {
    Pick(usize),     // 0..candidates.len()
    Skip,
    TimedOut,
}

#[derive(Debug, Clone)]
pub enum ReportableOutcome {
    Posted { chosen_index: usize, tweet_url: String },
    Skipped,
    TimedOut,
    GateRejected { chosen_index: usize, reason: String },
    PublishFailed { chosen_index: usize, reason: String },
}

#[derive(Debug, Error)]
pub enum ReviewDeliveryError {
    /// Bot connection failure / API error.
    #[error("delivery transport: {0}")]
    Transport(String),
    /// Timeout was reached before the user responded.
    /// (Returned ONLY if the impl chooses to surface timeout as an error
    /// rather than as `DeliveryOutcome::TimedOut`. Default: surface as
    /// outcome, not error.)
    #[error("timeout after {timeout_secs}s")]
    Timeout { timeout_secs: u64 },
    /// Pick callback was received but couldn't be parsed.
    #[error("invalid callback: {0}")]
    InvalidCallback(String),
}
```

### 3.3 heartbit-telegram extensions

```rust
// crates/heartbit-telegram/src/keyboard.rs

#[derive(Debug, Clone, PartialEq)]
pub enum CallbackAction {
    Approval { interaction_id: Uuid, decision: String },
    QuestionAnswer { interaction_id: Uuid, question_idx: usize, option_idx: usize },
    /// Persona-review pick — callback format: "p:{uuid}:{choice}"
    /// where choice is `0..9` (digit) or `skip`.
    PersonaPick { interaction_id: Uuid, choice: PickChoice },
}

#[derive(Debug, Clone, PartialEq)]
pub enum PickChoice {
    /// Pick a specific candidate by 0-based index.
    Index(usize),
    /// User pressed Skip.
    Skip,
}

/// Build inline keyboard markup for a persona review (N candidates + Skip).
pub fn persona_pick_buttons(interaction_id: Uuid, n: usize) -> Vec<(String, String)> {
    let id = interaction_id.to_string();
    let mut buttons = Vec::with_capacity(n + 1);
    for i in 0..n {
        buttons.push((
            format!("{}", i + 1),  // 1-based label for users
            format!("p:{id}:{i}"),
        ));
    }
    buttons.push(("Skip".into(), format!("p:{id}:skip")));
    buttons
}

// parse_callback_data adds:
//   Some(&"p") => parse "p:{uuid}:{choice}" → CallbackAction::PersonaPick
```

### 3.4 CLI wiring

```rust
// crates/heartbit-cli/src/persona.rs

#[derive(clap::Subcommand, Debug)]
pub enum PersonaCommand {
    Run {
        name: String,
        #[arg(long, value_name = "PROMPT")]
        once: String,
        /// Send candidates to Telegram for review and post on pick.
        /// Without this flag: runs P1.3c direct mode (judge picks, stdout).
        #[arg(long, default_value = "false")]
        review: bool,
    },
    // ...other subcommands unchanged
}
```

CLI dispatch branches on `review`:
- `false` → existing P1.3c `run_pipeline` path (unchanged)
- `true` → new `run_review_pipeline` path with `TelegramReviewDelivery` + `TwitterThreadTool` + env-based `CredentialResolver`

## 4. Telegram review delivery (CLI-side impl)

### 4.1 `TelegramReviewDelivery`

Lives in `crates/heartbit-cli/src/persona_review.rs` (new module). Owns a `teloxide::Bot` and a Tokio mutex of pending pick resolvers keyed by `interaction_id`.

```rust
pub struct TelegramReviewDelivery {
    bot: teloxide::Bot,
    chat_id: ChatId,
    timeout: Duration,
    /// Map of interaction_id → oneshot::Sender<DeliveryOutcome>
    pending: Arc<Mutex<HashMap<Uuid, oneshot::Sender<DeliveryOutcome>>>>,
    /// Background dispatcher task handle (created on first deliver call).
    dispatcher: OnceCell<JoinHandle<()>>,
}

impl TelegramReviewDelivery {
    pub fn from_env() -> Result<Self, ReviewDeliveryError> {
        let token = std::env::var("HEARTBIT_TELEGRAM_TOKEN")
            .map_err(|_| ReviewDeliveryError::Transport("HEARTBIT_TELEGRAM_TOKEN env var not set".into()))?;
        let chat_id: i64 = std::env::var("HEARTBIT_TELEGRAM_REVIEW_CHAT_ID")
            .map_err(|_| ReviewDeliveryError::Transport("HEARTBIT_TELEGRAM_REVIEW_CHAT_ID env var not set".into()))?
            .parse()
            .map_err(|e| ReviewDeliveryError::Transport(format!("invalid HEARTBIT_TELEGRAM_REVIEW_CHAT_ID: {e}")))?;
        let timeout_secs = std::env::var("HEARTBIT_REVIEW_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3600);  // 1 hour default
        // ...
    }
}
```

### 4.2 Send + await flow

In `deliver_and_await`:
1. Build the message text via `build_review_message(message)`.
2. Build inline keyboard via `persona_pick_buttons(interaction_id, candidates.len())`.
3. Lazily start a background task on first call: `bot.dispatcher()` long-poll loop with a callback handler. The handler parses callback data, looks up the interaction_id in `pending`, and resolves the oneshot.
4. `bot.send_message(chat_id, text).reply_markup(keyboard).await` — captures the returned `Message.id` for the receipt.
5. Insert `interaction_id → oneshot::Sender` into `pending`.
6. `tokio::select!` on `oneshot::Receiver` vs `tokio::time::sleep(timeout)`:
   - Receiver fires → `outcome = received` (`Pick(i)` or `Skip`).
   - Timer fires → remove interaction_id from pending; `outcome = TimedOut`.
7. Return `DeliveredReview { outcome, receipt: { chat_id, message_id } }`.

### 4.3 Report flow

In `report(receipt, outcome)`:
1. Parse `chat_id` and `message_id` from `receipt.data`.
2. Build the report text via `build_report_message(outcome)`.
3. `bot.edit_message_text(chat_id, message_id, text).reply_markup(empty).await` — clears the keyboard, shows the outcome.

**Telegram message format (text-only per Q4):**

```
🪶 Draft for {persona_name} — {topic}

1️⃣ {candidate 0 text}

2️⃣ {candidate 1 text}

3️⃣ {candidate 2 text}

Pick one, or Skip
```

Inline keyboard: `[1]` `[2]` `[3]` `[Skip]` (N+1 buttons in one row, or wrapped if too many).

**Report formats:**

```
✅ Posted draft 2 — https://x.com/<handle>/status/{tweet_id}

{chosen draft text}
```

```
❎ Skipped (no post)
```

```
⏰ Timed out — no pick within {seconds}s
```

```
🚫 Draft rejected by publish_gate: {reason}
```

```
⚠️ Publish failed: {reason}
```

### 4.4 Posting via `TwitterThreadTool` direct

In `run_review_pipeline`'s post-pick branch:

```rust
let chosen = &candidates[chosen_index];
check_publish_gate(&chosen.draft, &profile)
    .map_err(|e| ReviewError::Pipeline(PipelineError::PublishGate(e)))?;

let tweets = parse_thread_tweets(&chosen.draft);
let exec_ctx = ExecutionContext {
    credentials: Some(cfg.credentials.clone()),
    ..Default::default()
};
let input = serde_json::json!({"tweets": tweets});
let tool_output = cfg.twitter_tool.execute(&exec_ctx, input).await?;
// Parse tool_output for tweet IDs and URLs
```

`parse_thread_tweets`:

```rust
pub(crate) fn parse_thread_tweets(draft: &str) -> Vec<String> {
    draft
        .split("\n\n")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}
```

## 5. Error handling

`ReviewError` is the top-level error from `run_review_pipeline`. Three variants:
- `Pipeline` — wraps `PipelineError` (any error from candidate generation, snapshot load, etc.); uses `#[from]`.
- `Delivery` — wraps `ReviewDeliveryError` (transport/timeout-as-error/parse failures from the delivery layer); uses `#[from]`.
- `InvalidConfig` — config validation at run start.
- `Report` — non-fatal report failure (e.g., Telegram returns 400 on edit). The post may have succeeded; logged but doesn't fail the run.

**publish_gate failure** does NOT raise `ReviewError` — it produces `ReviewOutcome::GateRejected` (via `report()` to the user) and returns `Ok(ReviewOutput { outcome: GateRejected { ... }, ... })`. The user is informed via Telegram.

**Twitter API failure** similarly produces `ReviewOutcome::PublishFailed` (via `report()`) and returns `Ok(...)`.

**Skip / TimedOut** return `Ok(ReviewOutput { outcome: Skipped | TimedOut, ... })` — these are not errors.

This matches the umbrella spec §6.4: "A user `/skip` is **not** treated as a negative preference signal." Same applies to TimedOut.

## 6. Phase 0 enforcement

Per the umbrella spec §6.4, Phase 0 is "100% candidates → Telegram." In P1.3d this is the **only** mode that actually posts to X. There is no auto-publish path. The `--review` flag is the user's opt-in to posting; without it, the existing P1.3c direct mode runs (no posting, stdout only).

No phase-tracking state is added in P1.3d. P1.4 adds:
- Phase counter on persona TOML (or in PostgreSQL audit table)
- Phase advancement triggers (50 picks for 0→1, score-based for higher)
- Auto-publish path (Phase 1+ with confidence threshold)

For P1.3d, "Phase 0" is implicit: there's no auto-publish path to bypass.

## 7. Testing

| File | Tests | What they cover |
|---|---|---|
| `heartbit-telegram/src/keyboard.rs` | +6 unit | `persona_pick_buttons` shape (N=3 → 4 buttons; N=1 → 2 buttons; N=0 rejected); `parse_callback_data` round-trip for Pick + Skip; invalid format errors |
| `heartbit-ghost/src/review/delivery.rs` | +3 unit | `DeliveryOutcome` / `ReportableOutcome` / `ReviewDeliveryError` display sanity tests |
| `heartbit-ghost/src/review/prompts.rs` | +4 unit | `build_review_message` (3 candidates rendered with persona + topic header; emoji + per-candidate trim; max-length bounds; special chars escaped) |
| `heartbit-ghost/src/review/tweet_split.rs` | +4 unit | `parse_thread_tweets` (single → vec of 1; thread → split on \n\n; whitespace trimmed; empty entries filtered) |
| `heartbit-ghost/src/review/mod.rs` (integration) | +6 `#[tokio::test]` | (1) pick → publish_gate passes → twitter_tool fires → `ReviewOutcome::Posted`; (2) skip → `ReviewOutcome::Skipped`, no twitter_tool call; (3) timeout → `ReviewOutcome::TimedOut`, no twitter_tool call; (4) pick → publish_gate fails → `ReviewOutcome::GateRejected`, no twitter_tool call; (5) pick → publish_gate passes → twitter_tool returns Err → `ReviewOutcome::PublishFailed`; (6) Pipeline error during candidate generation → `ReviewError::Pipeline`. |
| `heartbit-cli/src/persona_review.rs` | +0 | TelegramReviewDelivery integration tests skipped (real teloxide requires bot + chat). Unit-test the message-format helpers if any are extracted. |

**MockReviewDelivery test helper** lives in `heartbit-ghost/src/review/mod.rs` test module:

```rust
struct MockReviewDelivery {
    /// Pre-canned outcome to return.
    outcome: DeliveryOutcome,
    /// Track the report() calls for assertions.
    reports: Mutex<Vec<ReportableOutcome>>,
}

#[async_trait]
impl ReviewDelivery for MockReviewDelivery { /* ... */ }
```

**MockTwitterTool test helper** — implements `Tool` trait, returns canned `ToolOutput::success(json!({"tweet_ids": ["123"], ...}))`. Tests can also wire a failure variant.

Total: ~23 net new tests. Workspace 3989 → ~4012.

## 8. ADs (architecture decisions)

| AD | Decision | Reason |
|---|---|---|
| AD-1 | New `run_review_pipeline` entry point, not extension of `run_pipeline` | Distinct lifecycles; preserves P1.3c contract; no breaking changes to existing tests |
| AD-2 | Skip the `publisher` LLM recipe; call `twitter_thread` tool directly | Saves an LLM call; publisher recipe value is just deterministic tool routing (single vs reply, and replies aren't in P1.3d scope); defer recipe path to P1.4 daemon-mode multi-tenant credential plumbing |
| AD-3 | Standalone teloxide bot in CLI process (not daemon-mode) | MVP scope; daemon path layers on top in P1.4 cron/mention triggers; standalone validates the end-to-end flow without touching daemon code |
| AD-4 | Skip image generation in review mode | `twitter_thread` doesn't accept media yet (P1.4); generating an image we can't attach is performative |
| AD-5 | Edit Telegram message in place on outcome | Cleanest UX; matches Q7; broken keyboards (re-clickable buttons after pick) avoided |
| AD-6 | Skip and TimedOut are non-fatal outcomes (not errors) | Matches umbrella spec §6.4 — silence ≠ rejection |
| AD-7 | publish_gate / twitter API failures produce outcomes (not errors) | User is informed via Telegram report; CLI exits cleanly with non-zero exit code (caller can distinguish) |
| AD-8 | Abstract Telegram delivery behind `ReviewDelivery` trait | Enables fast deterministic integration tests without teloxide; production uses `TelegramReviewDelivery`; heartbit-ghost stays free of teloxide dependency |
| AD-9 | Abstract Twitter posting behind `Arc<dyn Tool>` (not new trait) | Tool trait already exists; no new abstraction; tests inject mock Tool with canned ToolOutput |
| AD-10 | `ReviewConfig` is its own struct (not `PipelineConfig` extension) | Explicit composition; review-only fields (`delivery`, `twitter_tool`, `credentials`) are obvious; PipelineConfig stays unchanged |
| AD-11 | Pick correlation = single in-flight per CLI process, UUID-keyed | Standalone CLI = 1 review at a time; UUID-keyed map is overkill but matches the existing `interaction_id` pattern from heartbit-telegram and trivially extends to multi-flight if daemon-mode arrives |
| AD-12 | Timeout default = 1 hour, env override `HEARTBIT_REVIEW_TIMEOUT_SECS` | Reasonable for "morning ideas" workflow; user can override per-deployment |
| AD-13 | Bot token + chat_id from env vars (`HEARTBIT_TELEGRAM_TOKEN`, `HEARTBIT_TELEGRAM_REVIEW_CHAT_ID`) | Matches existing heartbit-telegram convention; no new config file |
| AD-14 | X API credentials from `CredentialResolver` (env-based default via `heartbit::auth::vault::CredentialResolver::env_only()`) | Reuses existing infrastructure; resolver chains env → vault as today |
| AD-15 | Phase 0 = `--review` is the only post-to-X path | No phase counter or advancement logic in P1.3d; P1.4 adds Phase 1+ |
| AD-16 | `parse_thread_tweets` splits on `\n\n` (matches `publish_gate`) | Same convention as P1.3b's publish_gate; consistent across the pipeline |

## 9. Acceptance criteria

P1.3d is done when:

1. All public types compile under `cargo check -p heartbit-ghost --all-features` and `cargo check -p heartbit-cli --all-features`.
2. `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green.
3. ~23 net new tests pass (6 keyboard + 3 delivery types + 4 prompts + 4 tweet_split + 6 integration).
4. New public surface from `heartbit_ghost::review`: `ReviewConfig`, `ReviewOutput`, `ReviewOutcome`, `ReviewError`, `run_review_pipeline`, `ReviewDelivery`, `ReviewMessage`, `DeliveryOutcome`, `ReportableOutcome`, `DeliveryReceipt`, `DeliveredReview`, `ReviewDeliveryError`, `parse_thread_tweets`, `build_review_message`.
5. New public surface from `heartbit_telegram`: `CallbackAction::PersonaPick`, `PickChoice`, `persona_pick_buttons`.
6. `heartbit persona run heartbit-ghost:x --once "<topic>" --review` runs end-to-end against:
   - real `OPENROUTER_API_KEY` for candidate generation
   - real `HEARTBIT_TELEGRAM_TOKEN` + `HEARTBIT_TELEGRAM_REVIEW_CHAT_ID` for the Telegram bot
   - real `HEARTBIT_X_API_KEY` etc. for the Twitter API (credentials format matches the existing X tool convention)
   - User receives a Telegram message with 3 candidates + inline keyboard. User picks → tweet posts. User skips → graceful exit. Timeout → graceful exit. (User-driven verification, same path as P1.3b/c acceptance §5.)

## 10. Out of scope (deferred)

- Persistent pick storage in memory namespace (Reflection entry per umbrella spec §6.3) → P1.3e
- Few-shot exemplar retrieval from past picks → P1.3e
- Daemon-mode review path (CLI submits → daemon handles bot + posting) → P1.4
- Phase 1+ autonomy (auto-publish path with confidence threshold) → P1.4
- Audit log integration → P1.4
- LLM-based content guardrails (PII / brand safety / electoral) → P1.4
- `twitter_thread_with_media` tool / image attachment to actual tweets → P1.4
- Reply triggers / mention polling → P1.4
- Cross-account anti-coordination guard → P1.4
- Engagement backfill (24h posted_engagement snapshot) → P1.4

## 11. Reference

- Umbrella spec: `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md` §3, §6, §6.2, §6.4
- P1.3a recipes (publisher recipe is documented but skipped per AD-2): `crates/heartbit-ghost/src/agents/publisher.rs`
- P1.3c spec (predecessor): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3c-multi-candidate-design.md`
- heartbit-telegram callback patterns: `crates/heartbit-telegram/src/keyboard.rs` (existing `Approval` and `QuestionAnswer` variants)
- ExecutionContext + CredentialResolver: `crates/heartbit-core/src/execution_context.rs`
- TwitterThreadTool: `crates/heartbit-ghost/src/tools/thread.rs`
- Existing CredentialResolver impl: `crates/heartbit/src/auth/vault.rs::CredentialResolver::env_only()`
