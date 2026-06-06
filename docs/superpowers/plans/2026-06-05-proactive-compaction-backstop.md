# Proactive Compaction Backstop (Leverage #2) — Plan

> **For agentic workers:** implement task-by-task with TDD (red→green→commit). Steps use `- [ ]`.

**Goal:** A token-aware proactive compaction *backstop* layered on top of #1's pruner: when even the *post-prune* request reaches a fraction (default 0.70) of the model's context window, summarize old context (reusing the existing `generate_summary`/`flush_to_memory`/`inject_summary`). Opt-in in core (caller supplies the window); on-by-default in the TUI.

**Design (approved):**
- Trigger on the REAL `usage.input_tokens` of the turn just sent (the pruner already rewrote `req.messages`, so this is the post-prune size — #1 and #2 compose, never fight).
- Window is supplied by the caller; core can't look it up. **When no window is set, keep the existing `summarize_threshold` estimate path unchanged** (never guess a window).
- Anti-thrash: never proactively compact two turns running.
- Justification is quality (attention/rot), not cost. One-turn-late by nature → 0.70 leaves headroom.
- Out of scope: leverage #3 (preservation schema), "append-only cache-aware" (impossible with prefix caching + shrinking).

**Gate per commit:** `cargo fmt --all -- --check && cargo clippy --workspace --exclude mini-crm --all-targets -- -D warnings && cargo test --workspace --exclude mini-crm`

---

## Task 1: core — builder/runner fields + setters (mechanical)

**Files:** `crates/heartbit-core/src/agent/runner.rs` (struct field + constructor default), `crates/heartbit-core/src/agent/builder.rs` (struct field + setters + build transfer).

- [ ] **Step 1: add the runner struct fields** — in `struct AgentRunner<P>` (near `summarize_threshold`):
```rust
    /// Model context window (tokens) for the proactive-compaction backstop. When
    /// `Some`, compaction triggers on real `usage.input_tokens` crossing
    /// `compaction_threshold_fraction * window`. `None` → fall back to the
    /// `summarize_threshold` estimate path.
    pub(super) context_window_tokens: Option<u32>,
    /// Fraction of the context window at which the backstop fires (default 0.70).
    pub(super) compaction_threshold_fraction: f32,
```
- [ ] **Step 2: builder struct fields + constructor defaults** — add the same two fields to `AgentRunnerBuilder` (builder.rs), and in the `AgentRunner::builder` constructor (runner.rs, where `summarize_threshold: None,` is) add:
```rust
            context_window_tokens: None,
            compaction_threshold_fraction: 0.70,
```
- [ ] **Step 3: builder setters** (builder.rs):
```rust
    /// Set the model context window (tokens) to enable the proactive compaction
    /// backstop (triggers on real prompt tokens crossing the threshold fraction).
    pub fn context_window_tokens(mut self, tokens: u32) -> Self {
        self.context_window_tokens = Some(tokens);
        self
    }

    /// Fraction of the context window at which to proactively compact (default 0.70).
    pub fn compaction_threshold_fraction(mut self, fraction: f32) -> Self {
        self.compaction_threshold_fraction = fraction;
        self
    }
```
- [ ] **Step 4: build() transfer** (builder.rs, in the `AgentRunner { ... }` literal):
```rust
            context_window_tokens: self.context_window_tokens,
            compaction_threshold_fraction: self.compaction_threshold_fraction,
```
- [ ] **Step 5:** `cargo build -p heartbit-core` clean → commit `feat(agent): builder knobs for proactive compaction backstop`.

---

## Task 2: core — real-token proactive trigger + anti-thrash guard

**Files:** `crates/heartbit-core/src/agent/runner.rs` (a pure helper + the proactive site at ~1760 + a loop-scoped guard flag). Tests in the same file.

- [ ] **Step 1: write the failing pure-helper test** — add to `runner.rs` `#[cfg(test)] mod tests`:
```rust
    #[test]
    fn over_window_fraction_triggers_at_or_above_budget() {
        // 70% of 1000 = 700.
        assert!(super::over_window_fraction(700, 1000, 0.70));
        assert!(super::over_window_fraction(800, 1000, 0.70));
        assert!(!super::over_window_fraction(699, 1000, 0.70));
        // window 0 never triggers (avoid div-by-zero / nonsense).
        assert!(!super::over_window_fraction(10, 0, 0.70));
    }
```
- [ ] **Step 2: run → FAIL** (`over_window_fraction` undefined). `cargo test -p heartbit-core --lib agent::runner::tests::over_window_fraction_triggers_at_or_above_budget`.
- [ ] **Step 3: add the pure helper** (free fn in runner.rs, module level):
```rust
/// Whether `input_tokens` has reached `fraction` of the context `window`.
/// Returns false for a zero window (unknown → no trigger).
fn over_window_fraction(input_tokens: u32, window: u32, fraction: f32) -> bool {
    window > 0 && input_tokens as f32 >= fraction * window as f32
}
```
- [ ] **Step 4: run → PASS.**
- [ ] **Step 5: capture the real input tokens per turn.** Find where the turn's LLM `response` is obtained (its `usage` is used at ~916). Add, right after the response is available:
```rust
                let last_input_tokens = response.usage.input_tokens;
```
(Use the actual binding name; `response.usage` is `Copy`.)
- [ ] **Step 6: add the loop-scoped anti-thrash flag.** Near the existing `let mut compacted_last_turn = false;` (~line 584), add:
```rust
                let mut proactive_compacted_last_turn = false;
```
- [ ] **Step 7: rewire the proactive trigger** at ~1760. Replace the current condition
```rust
                if !tool_interrupted
                    && let Some(threshold) = self.summarize_threshold
                    && ctx.message_count() > 5
                    && ctx.needs_compaction(threshold)
                {
```
with a trigger that prefers the real %-window path and keeps the estimate path as fallback, gated by the anti-thrash flag:
```rust
                // Proactive compaction backstop. Prefer the real post-prune token
                // count vs the window fraction; fall back to the chars/4 estimate
                // vs summarize_threshold when no window is known. Never compact two
                // turns running (anti-thrash).
                let want_proactive = ctx.message_count() > 5
                    && !proactive_compacted_last_turn
                    && match self.context_window_tokens {
                        Some(window) => over_window_fraction(
                            last_input_tokens,
                            window,
                            self.compaction_threshold_fraction,
                        ),
                        None => self
                            .summarize_threshold
                            .is_some_and(|t| ctx.needs_compaction(t)),
                    };
                if !tool_interrupted && want_proactive {
```
Inside the block, after `ctx.inject_summary(summary, 4);`, set the flag:
```rust
                        proactive_compacted_last_turn = true;
```
And ensure the flag is RESET to `false` on any turn that did NOT proactively compact, so it caps at "every other turn" rather than "once ever". The simplest correct shape: just before the `if !tool_interrupted && want_proactive {` line, capture `let did = !tool_interrupted && want_proactive;` then after the whole block do `proactive_compacted_last_turn = did;`. (Adjust so the flag is `true` only for the turn immediately after a proactive compaction.) Keep the existing `generate_summary`/`flush_to_memory_before_compaction(&ctx, 4)`/`ContextSummarized` emit unchanged.

> NOTE: preserve the existing behavior exactly when `context_window_tokens` is `None` AND `summarize_threshold` is set (the old estimate path) — an existing test may cover it; keep it green.

- [ ] **Step 8: write the integration tests** (runner.rs tests, using `MockProvider` whose responses carry a chosen `usage.input_tokens`):
```rust
    #[tokio::test]
    async fn proactive_compaction_fires_when_real_tokens_cross_window_fraction() {
        // window 1000, fraction 0.70 → budget 700. A turn reporting 800 input
        // tokens (and enough messages) must emit ContextSummarized.
        // Build a MockProvider: turn 1 = a tool call (to grow message_count > 5
        // via a couple of turns) then a final response with input_tokens >= 800.
        // Capture events; assert at least one AgentEvent::ContextSummarized.
        // ... (use the module's existing MockProvider + on_event capture pattern)
    }

    #[tokio::test]
    async fn proactive_compaction_does_not_fire_below_fraction() {
        // Same setup but input_tokens = 600 (< 700) → NO ContextSummarized.
    }

    #[tokio::test]
    async fn proactive_compaction_does_not_thrash_two_turns_running() {
        // Two consecutive turns both reporting input_tokens >= 800 → exactly ONE
        // ContextSummarized (the guard suppresses the second).
    }
```
Adapt to the real `MockProvider` API (set `usage.input_tokens` on the responses; reuse the `on_event` capture from existing tests like `on_event_emits_tool_call_events`). Ensure `message_count() > 5` by issuing enough turns. If building these precisely is hard, at minimum ship the first two; the thrash test is the most valuable — keep it if feasible. Report what you built.
- [ ] **Step 9: run the tests → all green;** run the existing summarization tests too (`cargo test -p heartbit-core --lib agent::runner` and `agent::context`) to confirm no regression in the estimate path.
- [ ] **Step 10:** FULL workspace gate → commit `feat(agent): real-token proactive compaction backstop + anti-thrash guard`.

---

## Task 3: TUI — enable the backstop by default (single-agent)

**Files:** `crates/heartbit-tui/src/main.rs` (thread the window into the single-agent builder), `crates/heartbit-tui/src/app.rs` already has `context_limit()`.

- [ ] **Step 1:** in `spawn_agent`, capture the window from the catalog:
```rust
    let context_window = app.context_limit().map(|w| w.min(u32::MAX as u64) as u32);
```
and pass it as a new `build_engine` arg (add `context_window: Option<u32>` to the signature, threaded through the call).
- [ ] **Step 2:** in `build_engine`, on the SINGLE-agent builder (where `.context_recall_store(...)` is wired), add — only when the window is known:
```rust
        if let Some(window) = context_window {
            rb = rb.context_window_tokens(window); // fraction defaults to 0.70
        }
```
(Leave the fraction at the 0.70 default. When `context_window` is `None` — catalog not loaded yet — the backstop simply doesn't engage this spawn; it engages on the next (re)spawn once the window is known. Document this best-effort timing in a code comment.)
- [ ] **Step 3:** `cargo build -p heartbit-tui` clean; `cargo test -p heartbit-tui` green.
- [ ] **Step 4:** FULL workspace gate → commit `feat(tui): enable proactive compaction backstop (70% window, single-agent)`.

---

## Known limitation (document, don't fix here)
The TUI captures the window at spawn; if the OpenRouter catalog hasn't loaded yet (eager startup spawn), the backstop engages on the next (re)spawn (e.g. after `/model`). A follow-up could re-spawn the idle agent when `Msg::ModelsLoaded` first provides the window. Out of scope for this plan.
