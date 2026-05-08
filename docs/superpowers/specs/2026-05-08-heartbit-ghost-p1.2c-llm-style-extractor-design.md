# heartbit-ghost P1.2c — LLM style extractor design

**Status:** approved 2026-05-08
**Branch:** `feat/heartbit-ghost-p1.2c`
**Predecessors:** P1.2a (style profile schema), P1.2b (corpus storage). Both merged to `main`.
**Successors:** P1.2d (blend algorithm), P1.2e (CLI bodies), P1.4 (runtime conditioning).

## 1. Goal

Given a writer's reference corpus (`heartbit_ghost::corpus::Corpus`), call an LLM to produce a structured `StyleProfile` (`heartbit_ghost::voice::StyleProfile`) describing that writer's voice — the analyst-persona LLM call from umbrella spec §2.3.

The library surface needs to be enough that:

- P1.2d's blend algorithm can call `extractor.extract(&corpus).await` for each writer in a `BlendRecipe` to produce per-writer profiles before merging
- P1.2e's `heartbit persona profile rebuild` CLI body can wire one extraction per registered writer

Out of scope for this phase: blending profiles into one (P1.2d), CLI wiring (P1.2e), runtime conditioning of the writer agent (P1.4), embedding generation (P1.4), TOML serialization to disk (P1.2e).

## 2. Architecture

A single new file inside the existing `voice/` module:

```
crates/heartbit-ghost/src/voice/
├── mod.rs          # add re-exports
├── style.rs        # P1.2a — schema (untouched)
├── blend.rs        # P1.2a — recipe + partial (untouched)
├── error.rs        # P1.2a — VoiceError (untouched)
└── extractor.rs    # NEW (P1.2c) — LLM call + sampling + parse + validate
```

Voice owns all `StyleProfile`-related concerns. The extractor produces a `StyleProfile`, so it lives in `voice/`. Its error type is separate from `VoiceError` (different concern: LLM/network/parse failures, not schema validation).

**Provider injection** — `Arc<BoxedProvider>`, mirroring `LlmJudgeGuardrail` (the established single-shot pattern in `crates/heartbit-core/src/agent/guardrails/llm_judge.rs`). Caller decides whether to wrap with `RetryingProvider` (HTTP retries) and/or `CascadingProvider` (cost-tier escalation) — those concerns compose at the construction site, not inside the extractor.

**Dependencies** — `heartbit_core::llm::{BoxedProvider, LlmProvider, CompletionRequest, CompletionResponse, ContentBlock, ...}`, `serde_json`, `thiserror`, `tokio::time::timeout`, `chrono` (already a workspace dep from P1.2b). No new workspace deps.

## 3. Public API

```rust
// in heartbit-ghost::voice::extractor

pub struct StyleExtractor { /* ... */ }
pub struct StyleExtractorBuilder { /* ... */ }

impl StyleExtractor {
    /// Start building an extractor with the supplied LLM provider.
    pub fn builder(provider: Arc<BoxedProvider>) -> StyleExtractorBuilder;

    /// Extract a `StyleProfile` from `corpus`.
    ///
    /// Selects the top `sample_size` entries by engagement (likes desc,
    /// then reposts, then replies, then posted_at desc), renders them
    /// into the user message, calls the LLM, parses the response as JSON,
    /// and runs `StyleProfile::validate()`.
    pub async fn extract(&self, corpus: &Corpus) -> Result<StyleProfile, ExtractError>;
}

impl StyleExtractorBuilder {
    pub fn sample_size(self, k: usize) -> Self;
    pub fn max_response_tokens(self, t: u32) -> Self;
    pub fn timeout(self, d: Duration) -> Self;
    pub fn system_prompt(self, prompt: impl Into<String>) -> Self;
    pub fn build(self) -> StyleExtractor;
}

/// The default analyst-persona system prompt. Public so callers can
/// inspect or wrap it before passing back via `system_prompt()`.
pub fn default_system_prompt() -> &'static str;
```

Re-exports added to `voice/mod.rs`:

```rust
pub use extractor::{
    ExtractError, StyleExtractor, StyleExtractorBuilder, default_system_prompt,
};
```

**Builder defaults:**

| Field | Default |
|-------|---------|
| `sample_size` | 50 |
| `max_response_tokens` | 2048 |
| `timeout` | 60 seconds |
| `system_prompt` | `default_system_prompt()` (see §5) |

`StyleExtractor` is not `Default` (no sensible default for `provider`).

**Single entry point: only `extract`.** No `extract_with_overrides`, `extract_to_toml`, or `extract_streaming`. Override merging (P1.2d) and TOML serialization (P1.2e) compose at the call site, not inside the extractor.

## 4. Data types

```rust
/// Configuration for the style extractor (built via builder).
pub struct StyleExtractor {
    provider: Arc<BoxedProvider>,
    sample_size: usize,
    max_response_tokens: u32,
    timeout: Duration,
    system_prompt: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    #[error("corpus is empty for writer '{0}'")]
    EmptyCorpus(String),

    #[error("llm: {0}")]
    Llm(#[source] heartbit_core::Error),

    #[error("llm call timed out after {0:?}")]
    Timeout(Duration),

    #[error("llm produced no text response")]
    EmptyResponse,

    /// JSON parse failure. `raw` carries the offending output.
    #[error("json parse: {source}")]
    JsonParse {
        #[source]
        source: serde_json::Error,
        raw: String,
    },

    /// Parsed cleanly but failed `StyleProfile::validate()` (sums, ranges, etc.).
    /// `raw` carries the offending output; `inner` is the validation error.
    #[error("validation: {inner}")]
    Validation {
        #[source]
        inner: VoiceError,
        raw: String,
    },
}
```

**Field decisions:**

- `sample_size: usize` — top-K cap, default 50. `extract` may be called with corpora smaller than K; that is fine.
- `max_response_tokens: u32` default 2048 — JSON for a full `StyleProfile` is ~1.2k tokens (16 fields, closed-vocab strings, small arrays); 2048 is comfortable.
- `timeout: Duration` default 60s — single LLM call. `tokio::time::timeout` wraps `provider.complete()`.
- `system_prompt: String` — pre-rendered at builder time so `extract` is hot-path-free of allocation.
- `Llm(#[source] heartbit_core::Error)` wraps rather than re-types provider errors. P0/P1.0 already supplies `error_class::classify` for retry/auth/rate-limit handling at the wrapper layer.
- Both `JsonParse` and `Validation` always carry the raw LLM output. Debuggability during prompt iteration is the load-bearing pain point.

## 5. The prompt

System prompt is hand-written, static, embedded as a `&'static str` constant. ~1.5k tokens. Schema enums listed inline with their wire format (matching what `serde(rename_all = "snake_case")` produces) so the LLM doesn't have to invent vocabulary.

```rust
const DEFAULT_SYSTEM_PROMPT: &str = r#"You are a writing-style analyst. Your job: read sample posts from one writer and produce a structured fingerprint of their voice as a JSON object.

OUTPUT FORMAT — emit a single JSON object matching exactly this shape (no preamble, no markdown fences, no commentary):

{
  "version": 1,
  "sentence_length_target": "short" | "mixed" | "long",
  "sentence_length_distribution": [<u8>, <u8>, <u8>, <u8>],   // % of posts at lengths [<10, 10–20, 20–40, >40] words. Must sum to 100.
  "fragment_frequency": "rare" | "occasional" | "common",
  "opening_patterns": [<one or more of: "claim_first" | "number_first" | "scene_first" | "question_first" | "aphoristic_first" | "anecdote_first" | "contrarian_first">],
  "opening_pattern_weights": [<f64>, ...],   // parallel to opening_patterns; in [0, 1]; must sum to 1.0
  "formatting": {
    "lowercase": <bool>,
    "periods": "always" | "optional" | "rare",
    "em_dashes": "preferred" | "ok" | "forbidden",
    "quotation_marks": "double" | "single" | "smart",
    "line_breaks": "single" | "double" | "rhythmic"
  },
  "emoji_policy": "never" | "rare_punchline_only" | "occasional" | "frequent",
  "hashtag_policy": "never" | "rare" | "topic_relevant" | "always",
  "specificity_target": "low" | "medium" | "high",
  "voice_traits": [<short snake_case strings>],
  "ai_tells_to_avoid": [<short strings the writer never uses>],
  "thread_rhythm": "linear" | "list_then_payoff" | "punchline_callbacks",
  "thread_max_length": <u32 in 1..=25>,
  "thread_opener_must_hook": <bool>,
  "topical_obsessions": [<short strings>],
  "topical_avoidances": [<short strings>]
}

ANALYSIS GUIDANCE
- Read every post before answering. Look for stable patterns, not one-off quirks.
- Prefer evidence-based claims: if you see 8 short sentences and 2 long ones, "short" is the target with a [60, 30, 10, 0]-ish distribution — not "mixed".
- voice_traits and ai_tells_to_avoid must be observed in the corpus. Do not invent generic AI advice.
- topical_obsessions/avoidances reflect what THIS writer actually posts about (or pointedly doesn't), not generic categories.
- If the writer mostly does standalone posts, set thread_max_length=1 and thread_opener_must_hook=false.

CONSTRAINTS — your JSON must satisfy these or it will be rejected:
- sentence_length_distribution sums to 100
- opening_patterns and opening_pattern_weights have the same length
- opening_pattern_weights sum to 1.0 (within 1e-6)
- thread_max_length is 1..=25
- enum strings match the snake_case vocabulary above exactly

OUTPUT THE JSON OBJECT ONLY. No "Here is the analysis", no code fences, no trailing prose.
"#;
```

**User message rendering** (`render_user_message`):

```text
Writer: @karpathy
Sample size: 47 posts (top by engagement)

POST 1 (1234 likes, 87 reposts, 12 replies):
the bitter lesson keeps winning. compute + scale + simple objective beats clever priors. again.

POST 2 (982 likes, 45 reposts, 3 replies):
if your benchmark needs explanation, your benchmark is the problem.

POST 3 (no engagement data):
i was at the lab yesterday and...

...

Now produce the JSON object.
```

Engagement-less entries render as `(no engagement data)`. `posted_at` is omitted from the message (not load-bearing for style). `tags` and `embedding` are also omitted in v0.1 (no signal value at this stage).

**Why static, not generated**: the schema is small (16 fields). LLM quality depends on hand-tuned guidance + explicit constraints, not on auto-generated JSON Schema (which is verbose and harder for the LLM to follow). Same logic that applies to `LlmJudgeGuardrail`'s static system prompt.

**Why no few-shot examples in v0.1**: a single demonstration would bias the LLM toward that profile; multi-example demonstrations are expensive to maintain. Schema + constraints + analysis guidance is enough for a Sonnet-class or GPT-4-class model. P1.4 can add 1–2 generic exemplars if real-world failure rates demand it.

## 6. Sampling logic

Pure deterministic function over `&[CorpusEntry]`:

```rust
fn select_top_k<'a>(entries: &'a [CorpusEntry], k: usize) -> Vec<&'a CorpusEntry> {
    let mut sorted: Vec<&CorpusEntry> = entries.iter().collect();
    sorted.sort_by(|a, b| {
        let a_eng = a.engagement.unwrap_or_default();
        let b_eng = b.engagement.unwrap_or_default();
        b_eng.likes.cmp(&a_eng.likes)
            .then(b_eng.reposts.cmp(&a_eng.reposts))
            .then(b_eng.replies.cmp(&a_eng.replies))
            .then_with(|| match (b.posted_at, a.posted_at) {
                (Some(b), Some(a)) => b.cmp(&a),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            })
    });
    sorted.truncate(k);
    sorted
}
```

Stable sort: equal-engagement entries with no `posted_at` preserve corpus load order. Pure function — directly testable without I/O.

## 7. Error handling

**Empty corpus** — `extract` short-circuits at the top: `if corpus.is_empty() { return Err(ExtractError::EmptyCorpus(corpus.writer().to_string())); }`. The provider is never called.

**Provider error** — `provider.complete(request)` returning `Err(heartbit_core::Error)` propagates as `ExtractError::Llm(...)`. The wrapped error preserves P1.0's classification (retryable / auth / rate-limit / etc.).

**Timeout** — the `complete()` future is wrapped in `tokio::time::timeout(self.timeout, ...)`. On elapsed, returns `ExtractError::Timeout(self.timeout)`. The in-flight HTTP request is cancelled by the underlying client when the future drops.

**Empty response** — if the response's `content` has no `ContentBlock::Text` entries (refusal, model produced only tool calls, etc.), returns `ExtractError::EmptyResponse`.

**Defensive markdown-fence stripping** — concatenate all `Text` blocks, trim. If the result starts with ` ```json ` (or ` ``` `) and ends with ` ``` `, strip a single fence pair before parsing. Common LLM failure mode despite the prompt saying not to fence; cheap to handle.

**JSON parse failure** — `serde_json::from_str::<StyleProfile>(&text)` returning `Err` → `ExtractError::JsonParse { source, raw: text }`.

**Validation failure** — `parsed.validate()` returning `Err(VoiceError)` → `ExtractError::Validation { inner: VoiceError, raw: text }`.

**No retry** — single call, single outcome. The caller wraps with `RetryingProvider` for transient HTTP errors (429/5xx surface in `ExtractError::Llm`); for parse/validate failures the caller decides whether to retry with the raw output as feedback (cheap to do at the call site).

**Concurrency / cancellation** — `extract` is `async`; if the caller's future is dropped, the timeout future is dropped, the in-flight request is cancelled at the network layer. No state to clean up.

## 8. Testing

~19 tests, all in-tree (`#[cfg(test)] mod tests` in `voice/extractor.rs`). Test helper: a hand-rolled `MockProvider` (~25 lines) implementing `LlmProvider` directly, returning canned text via `Mutex<Result<String, ...>>` and capturing the request that was sent.

**Pure logic — `select_top_k` (6 tests):**

- `select_top_k_empty_returns_empty`
- `select_top_k_smaller_than_k_returns_all`
- `select_top_k_orders_by_likes_desc`
- `select_top_k_tiebreaks_by_reposts_then_replies`
- `select_top_k_tiebreaks_by_posted_at_desc_when_engagement_equal`
- `select_top_k_engagementless_entries_sort_to_bottom`

**Builder defaults (1 test):**

- `builder_defaults_match_documented_values` — sample_size=50, max_response_tokens=2048, timeout=60s

**Happy path (2 tests):**

- `extract_returns_validated_profile_on_valid_json`
- `extract_strips_markdown_fences_around_json` — defensive against ```json … ``` wrappers

**Error paths (6 tests):**

- `extract_empty_corpus_returns_empty_corpus_error` — provider never called
- `extract_propagates_llm_provider_error`
- `extract_returns_timeout_when_provider_exceeds_budget` — mock has 200ms delay, extractor timeout=50ms
- `extract_returns_empty_response_when_no_text_blocks`
- `extract_invalid_json_returns_jsonparse_with_raw`
- `extract_invalid_profile_returns_validation_with_raw` — mock returns JSON where `sentence_length_distribution = [40, 40, 40, 40]` (sums to 160); assert `ExtractError::Validation` containing the schema validation message and `.raw` carrying the offending JSON

**Prompt rendering (3 tests):**

- `render_user_message_includes_writer_handle`
- `render_user_message_renders_engagement_when_present`
- `render_user_message_marks_engagementless_entries`

**Provider sees the right request (1 test):**

- `extract_passes_system_prompt_and_user_message_to_provider` — capture the request via the mock; assert system prompt is the default (or custom if overridden), user message contains the rendered corpus block, `max_tokens` matches builder.

**Quality gate** (mirrors P1.2a, P1.2b):

```bash
cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
```

Workspace test count: 3849 → ~3868.

## 9. Architecture decisions (ADs)

**AD-1 — JSON output, not TOML or tool-use.** LLMs are strongest at JSON. `StyleProfile` already implements `Serialize`/`Deserialize`; serde is format-agnostic, so the same Rust struct round-trips cleanly through JSON. Closed-vocab enums (`serde(rename_all = "snake_case")`) emit identical wire format in JSON and TOML, so no schema duplication. Tool-use / function-call would tie us to provider-specific machinery for marginal reliability gain.

**AD-2 — Top-K by engagement, default K=50, configurable.** Predictable, deterministic, testable. K=50 covers ~14k tokens at ~280 chars/post — well under any context limit and enough samples for robust style extraction. Engagement-less entries sort to the bottom (treated as zero engagement). Builder method `sample_size(k)` overrides.

**AD-3 — No retry; return Err with raw output.** Single call, single outcome. `ExtractError::JsonParse` and `ExtractError::Validation` carry the raw LLM text + the underlying error so the caller can debug or implement their own retry. Avoids hidden state and double-billing on flaky models. The caller can layer retry trivially: cheap to write at the call site, expensive to undo if baked in.

**AD-4 — `Arc<BoxedProvider>` injection, not generic provider.** Matches `LlmJudgeGuardrail`. Lets callers swap models (cheap analyst tier vs. expensive one) without recompiling. Composes with `RetryingProvider`/`CascadingProvider` at construction time.

**AD-5 — Static, hand-written system prompt; no schemars / autogen.** Schema is small (16 fields). Hand-tuned prompt with closed-enum vocabulary lists and analysis guidance outperforms autogenerated JSON Schema for prompt quality. Same pattern as `LlmJudgeGuardrail`.

**AD-6 — Defensive markdown-fence stripping.** Cheap to implement (~5 lines), substantially more robust against the most common "LLM ignored the no-fences instruction" failure mode. Strip exactly one fence pair; if the LLM emits something more exotic, that's a real prompt-tuning issue, not a wrapper problem.

**AD-7 — Single entry point: `extract`.** No `extract_with_overrides`, `extract_to_toml`, or `extract_streaming`. Override merging is P1.2d's concern; TOML serialization is P1.2e's; streaming has no use case for a JSON-emitting single-shot call.

## 10. Acceptance criteria

P1.2c is done when:

- All public types compile cleanly under `cargo check -p heartbit-ghost --all-features`
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green
- ~19 extractor tests pass; coverage spans pure sampling logic (6), builder defaults (1), happy path (2), all error paths (6), prompt rendering (3), and request shape (1)
- `heartbit_ghost::voice::{StyleExtractor, StyleExtractorBuilder, ExtractError, default_system_prompt}` are reachable as public surface
- A test verifies the extractor passes the configured system prompt + the rendered user message + the configured `max_response_tokens` to the provider

## 11. Out of scope (re-stated)

- Blend algorithm — merging N profiles into one (P1.2d)
- CLI bodies for `profile rebuild` and `profile diff` (P1.2e)
- Runtime conditioning of the writer agent (P1.4)
- Embedding generation (P1.4)
- TOML serialization to disk (P1.2e — the extractor returns a `StyleProfile` value; persistence is the caller's concern)
- Auto-retry on parse/validate failure (caller can layer this trivially using the carried `raw` field)
- Few-shot prompt examples (P1.4 if real-world failure rates demand it)
- Live LLM integration tests — the test suite is mock-only; live tests live at the call site (e.g., a P1.2e smoke example with a real key)
- Streaming output (no use case for a single-shot JSON-emitting call)
- Profile versioning beyond v=1 (P1.2a's deferred concern; revisit at first breaking schema change)

## 12. Reference

- Umbrella heartbit-ghost spec §2.3 (style extractor): `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md`
- P1.2a spec (style profile schema): `docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.2a-style-profile-schema-design.md`
- P1.2b spec (corpus storage): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2b-corpus-storage-design.md`
- LlmJudgeGuardrail (single-shot pattern reference): `crates/heartbit-core/src/agent/guardrails/llm_judge.rs`
- LlmProvider trait: `crates/heartbit-core/src/llm/mod.rs`
- Foundation: `docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md`
