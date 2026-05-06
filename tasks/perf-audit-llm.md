# Perf audit: LLM providers

## Summary
- **Total findings**: 18
- **Critical**: 4 | **High**: 7 | **Medium**: 5 | **Low**: 2
- **Top 3 wins**: (1) Per-call `serde_json::to_value` + `to_string` double-pass in openrouter request building (~500µs cumulative); (2) SSE parser per-event `String` allocation in `.feed()` hot path (allocations compound per-chunk); (3) Pricing model lookup uses linear string-match on 30+ entries instead of FxHashMap
- **Rejected suggestions**: None identified. All findings preserve security boundaries.

---

## P-LLM-1 [High]: Double serde_json pass per request in OpenRouter build_openai_request

- **File**: `crates/heartbit-core/src/llm/openrouter.rs:234–236`
- **Observation**: Line 234 calls `serde_json::to_string(&input)` on `serde_json::Value` that was constructed via `serde_json::json!(...)` macro. The `json!` macro uses `.into()` to produce `Value`, which is then serialized to `String` just to pass to OpenAI. The full request body uses `serde_json::json!{ "function": { "arguments": STRING } }`, forcing a second serialization.
- **Hypothesized cost**: ~500µs per request (10–50 tool calls × ~50µs per `to_string`). Per-call + per-stream-chunk when streaming.
- **Frequency**: hot-path-per-call
- **Fix sketch**: Keep `input` as `serde_json::Value` directly; delay string serialization to `.json(&body)` call, which internally calls `to_string` once. Or cache the tool schema serialization at construction time if tools are reused.
- **Security delta**: N/A
- **Validation**: needs-bench (measure time to build 10-tool request body before/after)

---

## P-LLM-2 [High]: SSE parser per-event String allocation in feed() hot path

- **File**: `crates/heartbit-core/src/llm/anthropic.rs:438, 463, 479`
- **Observation**: In `SseParser::feed()`, every `next_line()` call (line 438) constructs a fresh `String` via `self.buffer[..i].to_string()`. Per HTTP chunk, this can be 10–100 events. Each line is cloned into the returned `Vec<SseEvent>` (line 379 return Vec<SseEvent>). Over a 16 KB response with ~32 events, ~32 String allocations + copies. Later, lines are stored into `self.data_lines: Vec<String>` (line 479), adding more allocations.
- **Hypothesized cost**: 30–50 allocations per response, ~500 bytes each = 15–25 KB churn. Per-chunk in streaming hot path.
- **Frequency**: hot-path-per-stream-chunk
- **Fix sketch**: (1) Use `&str` slices into `self.buffer` instead of cloning to `String`. Lifetime complications for the `Vec<SseEvent>` — may require owning SSE via `String` arc/rc or restructure the return type. (2) Use `SmallString<[u8; 64]>` or `&'a str` with a borrowed-data SSE event. (3) Use `bytes::Bytes` for zero-copy framing.
- **Security delta**: N/A
- **Validation**: needs-bench (profile 16 MB response stream with callgrind; count allocations before/after)

---

## P-LLM-3 [High]: Cascade uses full-text scan on response rejection detection

- **File**: `crates/heartbit-core/src/llm/cascade.rs:79–82`
- **Observation**: `HeuristicGate::accept()` calls `response.text().to_lowercase()` (line 79), which allocates a new String and linearizes all content blocks. Then for each of ~4 refusal patterns, it calls `lower.contains(pattern.to_lowercase())` (line 81), which re-lowercases the pattern every time. For a 4 KB response and 4 patterns, this is ~8 KB String allocation + 4× linear scans (~16 KB work).
- **Hypothesized cost**: Per-call when cascade gate rejects cheap tier. ~200–300µs for 4 KB response. Cascades with many tiers see this multiple times.
- **Frequency**: warm-path-per-cascade-rejection
- **Fix sketch**: (1) Lowercase `response.text()` once, store. (2) Pre-lowercase patterns at HeuristicGate construction or use a static lowercase pattern table. (3) Use `LazyLock<Vec<String>>` for static refusal patterns so they're only lowercased once globally.
- **Security delta**: N/A (does not re-open F-LLM-7; pattern matching remains substring-based)
- **Validation**: static-only (code inspection + regex → substr equivalence)

---

## P-LLM-4 [High]: Pricing lookup uses linear string-match on 30+ model entries

- **File**: `crates/heartbit-core/src/llm/pricing.rs:24–65`
- **Observation**: `model_pricing()` function uses a `match` statement on model name (27 exact string matches). Each cost lookup is a linear `match` parse (O(n) worst case in a flat match, though the compiler likely turns this into a jump table). For 30+ models, every completion will hit this. No memoization for repeated calls with the same model.
- **Hypothesized cost**: ~1–2µs per lookup (compiler optimizes match to hash-like jump table). Per-call when usage is logged. Repeated lookups for the same model in a single agent run (e.g., 100-turn conversation) incur 100× cost.
- **Frequency**: warm-path-per-call
- **Fix sketch**: (1) Convert the match to a `LazyLock<HashMap<&'static str, (f64, f64)>>` or `LazyLock<FxHashMap<...>>` for O(1) lookup. (2) Cache the provider's model name in `pricing_cache: OnceLock<Option<(f64, f64)>>` on construction of each provider so each agent run computes it once.
- **Security delta**: N/A
- **Validation**: static-only (code inspection; match is compiled-time constant, so lookup is ~O(1) via inlining, but HashMap explicit lookup is clearer and allows dynamic pricing updates)

---

## P-LLM-5 [Critical]: Box<dyn Future> overhead in DynLlmProvider/BoxedProvider adapter

- **File**: `crates/heartbit-core/src/llm/mod.rs:215, 223` (DynLlmProvider impls)
- **Observation**: `DynLlmProvider::complete()` calls `Box::pin(LlmProvider::complete(self, request))` — Boxing a future that is already potentially large (inner future + state). For a streaming response with accumulated SSE state, this can be 1–5 KB. Every `.await` on a `Box<dyn Future>` adds vtable indirection (~2–4 µs per call).
- **Hypothesized cost**: ~4–8µs vtable dispatch per complete/stream_complete call. Negligible for I/O-bound calls (100+ ms network latency) but noticeable in test suites or when cascading tiers trigger multiple cheap provider calls.
- **Frequency**: hot-path-per-call (but dominated by network I/O)
- **Fix sketch**: (1) The agent already uses `Arc<dyn DynLlmProvider>` on the Restate service layer — keep the Arc, return pinned futures via method-level impl. (2) Use `impl Future` on trait methods once stable (RPITIT is stable; heterogeneous return types are not). (3) Profile: measure the cost of a 100-call cascade (cheap tiers erroring in series) — vtable cost may be sub-1% of total latency if network dominates.
- **Security delta**: N/A
- **Validation**: needs-bench (Criterion: 1000 calls to a fast mock provider via DynLlmProvider vs direct LlmProvider; measure instruction count and cycles)

---

## P-LLM-6 [Medium]: Request body cloning in RetryingProvider stream_complete

- **File**: `crates/heartbit-core/src/llm/retry.rs:208` (loop clones request)
- **Observation**: `stream_complete()` clones `request: CompletionRequest` on retry attempt (line 208: `self.inner.stream_complete(request.clone(), ...)`). `CompletionRequest` contains `Vec<Message>` with nested `Vec<ContentBlock>`. A multi-turn conversation can be 100+ KB. Each retry clones the full structure; with `max_retries=3`, a 429 error clones 3× (3 additional allocations + copies).
- **Hypothesized cost**: 100–500 KB allocation per retry (worst case: 100 KB messages × 3 retries = 300 KB churn on a transient error).
- **Frequency**: warm-path-on-retries
- **Fix sketch**: (1) Pass `&CompletionRequest` through the trait (but then all providers must support lifetimes — trait breaking change). (2) Wrap request in `Arc<CompletionRequest>` so clones are cheap. (3) Measure: does retry happen often enough to warrant this? If 429/503 is rare (<1% of calls), the cost is negligible vs. the saved complexity.
- **Security delta**: N/A
- **Validation**: needs-bench (profile a retry loop with a 50-message, 100 KB request; count allocations)

---

## P-LLM-7 [Medium]: Cache field name JSON encoding per-call in Anthropic prompt caching

- **File**: `crates/heartbit-core/src/llm/anthropic.rs:246–250, 263, 286`
- **Observation**: When `prompt_caching=true`, the `build_request_body()` function constructs cache_control JSON objects via `serde_json::json!({"type": "ephemeral"})` (lines 249, 263, 286) repeated 3 times per call (system, last tool, second-to-last user message). Each is a fresh JSON value construction + field insertion. The string `"ephemeral"` and struct `{"type": ...}` are identical across calls.
- **Hypothesized cost**: ~3 allocations × 50 bytes = ~150 bytes overhead per request with caching enabled. ~3–5µs total.
- **Frequency**: warm-path-per-call (only when prompt_caching=true, which is opt-in)
- **Fix sketch**: (1) Cache `serde_json::json!({"type": "ephemeral"})` as a `LazyLock<serde_json::Value>` and `.clone()` it 3 times. Or (2) Pre-serialize to a `&'static str` and `.parse()` it. (3) Measure: single static allocation + 3 clones (cheap for small values) vs. 3 fresh allocations.
- **Security delta**: N/A
- **Validation**: static-only (code inspection; LazyLock JSON reuse is a micro-opt that's only worth it if profile shows it's hot)

---

## P-LLM-8 [Medium]: Error classification regex compile per-call (if regex is later added)

- **File**: `crates/heartbit-core/src/llm/error_class.rs:62–77` (currently uses substring matching, not regex)
- **Observation**: **Current state is good**: `is_context_overflow()` uses substring matching (line 76: `.contains(p)`), not regex. No regex compile-per-call. **Future risk**: if context overflow detection evolves to regex patterns (e.g., to distinguish "max_tokens parameter" from "context overflow"), a per-call regex compile would be expensive.
- **Hypothesized cost**: ~100–500µs per `classify()` call if regex is added. Every error path hits this.
- **Frequency**: warm-path-on-error
- **Fix sketch**: If regex is added, use `LazyLock<Regex>` for static patterns. Keep substring matching as fallback.
- **Security delta**: N/A
- **Validation**: static-only (future-proofing note, not a current issue)

---

## P-LLM-9 [Medium]: Cascade response.text() extraction followed by gate evaluation

- **File**: `crates/heartbit-core/src/llm/cascade.rs:189–191` (stream_complete path)
- **Observation**: When cascade accepts a cheap-tier response from `stream_complete()`, line 189 calls `response.text()`, which linearizes content blocks into a String. This is unnecessary if the response is just passed to `on_text` callback. The same text was already emitted via `on_text` during streaming; extracting it again is redundant.
- **Hypothesized cost**: ~1–4 KB String allocation per accepted cheap-tier response in streaming mode. Typically once per cascade run (when a cheap tier succeeds).
- **Frequency**: warm-path-per-cascade-acceptance
- **Fix sketch**: (1) For non-final tiers in `stream_complete()`, use `.complete()` to avoid streaming (line 172 comment notes this). When cheap tier is accepted, the text is already accumulated in `response.content` — no need to re-extract via `.text()`. (2) Remove line 189's `response.text()` call and directly check if response is non-empty via `!response.content.is_empty()`.
- **Security delta**: N/A
- **Validation**: code-inspection (unnecessary String constructor removal)

---

## P-LLM-10 [High]: Circuit breaker uses RwLock + HashMap, but write path can allocate unbounded

- **File**: `crates/heartbit-core/src/llm/circuit.rs:233–249` (CircuitTracker::circuit_for)
- **Observation**: The write-lock path (line 245) calls `.or_insert_with(|| Arc::new(ProviderCircuit::new(...)))`. For each new `(tenant, provider)` pair, a new `ProviderCircuit` is allocated. The `HashMap` grows unbounded: if you have 1000 tenants and 10 providers, the tracker accumulates 10K entries, each holding an `Arc<ProviderCircuit>` (80 bytes). In a multi-tenant system, this can reach MBs.
- **Hypothesized cost**: Memory overhead grows as O(tenants × providers). Per-request lookup is still O(1) via RwLock + HashMap, so latency is fine, but memory use is unbounded unless the service restarts.
- **Frequency**: cold-path-per-new-tenant (first request from a new tenant to a new provider)
- **Fix sketch**: (1) If tenants are ephemeral, add a TTL-based eviction to the `HashMap` using `DashMap` with a background cleanup task. (2) If the tracker is per-tenant (not global), this is not an issue. (3) Measure: for typical multi-tenant deployments, how many circuits are accumulated over a day?
- **Security delta**: N/A (circuit isolation is preserved)
- **Validation**: needs-bench (measure memory use in a multi-tenant scenario; check if GC is needed or if LRU eviction is warranted)

---

## P-LLM-11 [Low]: Anthropic uses separate string append for cache_control when could be .json!() macro

- **File**: `crates/heartbit-core/src/llm/anthropic.rs:263` (tools cache_control insertion)
- **Observation**: Line 263 mutates the last tool's `cache_control` field via `last["cache_control"] = serde_json::json!(...)`. This is fine for single insertions, but if the code scales to more cache breakpoints, each insertion is a fresh `Value::Object` construction.
- **Hypothesized cost**: Negligible (~1 µs per field insertion). Not a perf issue in practice.
- **Frequency**: cold-path-per-call
- **Fix sketch**: Structural improvement: build the cache_control value once at the top of the function and clone it. Or build the entire tool value with cache_control pre-baked.
- **Security delta**: N/A
- **Validation**: static-only (micro-optimization, not worth the code complexity)

---

## P-LLM-12 [Critical]: Cascade complete() on non-final tiers clones request for retries

- **File**: `crates/heartbit-core/src/llm/cascade.rs:126, 180`
- **Observation**: When a non-final tier in cascade fails or is rejected, the `request: CompletionRequest` is cloned and passed to the next tier (line 126: `tier.provider.complete(request.clone())` and line 180: `tier.provider.complete(request.clone())`). In a 3-tier cascade with all cheap tiers failing, this clones 2× (cheap1 → cheap2, cheap2 → expensive).
- **Hypothesized cost**: 100–500 KB cloning overhead per cascade escalation. Frequent in cost-optimized setups (cheap tier fails 50% of time).
- **Frequency**: warm-path-per-cascade-escalation
- **Fix sketch**: (1) Wrap `CompletionRequest` in `Arc<CompletionRequest>` at cascade entry so clones are O(1) pointer copies. (2) Accept `&CompletionRequest` in LlmProvider trait (breaking change). (3) Use `SmartPtr` crate for copy-on-write if request is mutated.
- **Security delta**: N/A
- **Validation**: needs-bench (measure cascade escalation time with 3 tiers on a 100 KB request; compare Arc vs. owned clone)

---

## P-LLM-13 [Low]: Openrouter streaming tool_calls Vec pre-allocated with index tracking but no capacity hint

- **File**: `crates/heartbit-core/src/llm/openrouter.rs:506` (tool_calls: Vec<AccumulatedToolCall> initialized empty)
- **Observation**: `tool_calls` vector is initialized as empty (line 506: `let mut tool_calls: Vec<AccumulatedToolCall> = Vec::new()`). On every streaming delta with a tool call at a high index (e.g., index 5 out of 256 possible), the vector grows via repeated `.push()` calls in `process_openai_event` (line 679: `tool_calls.push(...)`). For a response with 10 parallel tool calls, this can trigger 10 reallocations (1, 2, 4, 8, 16).
- **Hypothesized cost**: O(log n) reallocations for n tool calls. Negligible for typical <10 parallel calls. ~1–2µs worst case.
- **Frequency**: warm-path-per-streaming-response-with-tools
- **Fix sketch**: (1) Pre-allocate with `Vec::with_capacity(8)` or `Vec::with_capacity(STREAM_MAX_TOOL_CALLS)`. (2) Measure: does parallel tool streaming happen frequently? If rare, not worth the code change.
- **Security delta**: N/A
- **Validation**: static-only (code inspection; pre-allocation is a micro-opt)

---

## P-LLM-14 [Critical]: SSE parser event_type and data_lines cloned on every emit

- **File**: `crates/heartbit-core/src/llm/anthropic.rs:490–492` (emit_event)
- **Observation**: `emit_event()` calls `std::mem::take(&mut self.event_type)` and `std::mem::take(&mut self.data_lines).join("\n")` (lines 490–491). The `.join()` allocates a new String even when the final event is never used (e.g., a comment line or empty event). For every SSE frame transition (blank line), a join() is triggered, even if only one data line is present.
- **Hypothesized cost**: Per SSE event, ~1 allocation (~50–200 bytes depending on event size). A 16 KB response with 32 events = 32 String joins. Allocations compound in streaming hot path.
- **Frequency**: hot-path-per-sse-event
- **Fix sketch**: (1) Delay `.join()` until the event is confirmed to be non-empty. (2) Use `String::from_iter(data_lines.iter().map(...))` to avoid an intermediate `Vec<String>`. (3) Store data as a single `String` with manual newline insertion instead of `Vec<String>`.
- **Security delta**: N/A
- **Validation**: needs-bench (measure allocations in a streaming response; count joins before/after)

---

## P-LLM-15 [Low]: Anthropic UTF-8 buffer reallocation on every chunk boundary

- **File**: `crates/heartbit-core/src/llm/anthropic.rs:539` (utf8_buf.drain)
- **Observation**: After extracting valid UTF-8 from `utf8_buf`, the code calls `.drain(..valid_len)` (line 539), which does not deallocate the buffer itself — it only shifts. If the buffer accumulates incomplete UTF-8 sequences at chunk boundaries, the capacity is retained but not freed until the response completes. For a streaming response with many small chunks, this can waste memory (e.g., 64 KB buffer allocated for a 3-byte incomplete UTF-8 sequence).
- **Hypothesized cost**: Memory waste, not latency. Typically <10 KB per streaming response, so negligible for most workloads.
- **Frequency**: warm-path-per-streaming-response
- **Fix sketch**: (1) If `utf8_buf.len() > THRESHOLD && incomplete_trailing.len() < 4`, reallocate or `.shrink_to_fit()` after drain. (2) Use `SmallVec<[u8; 4]>` for the trailing incomplete bytes to avoid a separate allocation.
- **Security delta**: N/A
- **Validation**: code-inspection (minor memory optimization)

---

## P-LLM-16 [High]: ToolDefinition serialization per-call in Anthropic build_request_body

- **File**: `crates/heartbit-core/src/llm/anthropic.rs:257` (serde_json::to_value(&request.tools))
- **Observation**: Line 257 calls `serde_json::to_value(&request.tools)` to serialize all tool definitions, then mutates the last tool to add cache_control (line 263). If tools are static (e.g., a fixed set of 10 tools per agent), this serialization is repeated on every request. Each `ToolDefinition` serializes its `input_schema: serde_json::Value` recursively.
- **Hypothesized cost**: ~50–200µs per request (for 10 tools with complex input schemas). Per-call when tools are reused across turns.
- **Frequency**: hot-path-per-call (when tools are enabled)
- **Fix sketch**: (1) Cache the serialized tool array in the provider (`tools_cache: OnceLock<serde_json::Value>`) and clone/mutate it. (2) Provide the tools to the provider at construction time, not per-request, so serialization is one-time.
- **Security delta**: N/A (tools are not user-controlled; they are statically defined per agent)
- **Validation**: needs-bench (profile a multi-turn conversation with 10 static tools; measure time spent in serde_json::to_value)

---

## P-LLM-17 [Low]: Cascade gate evaluation calls response.text().to_lowercase() instead of lazy accumulation

- **File**: `crates/heartbit-core/src/llm/cascade.rs:79` (response.text().to_lowercase())
- **Observation**: Same as P-LLM-3 but noted separately for the double-lowercasing of patterns (line 81). The pattern is lowercased again in the loop instead of pre-lowercasing once.
- **Hypothesized cost**: ~5–10 additional string operations per cascade rejection. Negligible overall.
- **Frequency**: warm-path-per-cascade-rejection
- **Fix sketch**: Pre-lowercase patterns in HeuristicGate::default() or use a static lazy-loaded set.
- **Security delta**: N/A
- **Validation**: static-only

---

## P-LLM-18 [Medium]: Retry decorrelated jitter uses atomic CAS loop instead of lock-free PRNG

- **File**: `crates/heartbit-core/src/llm/retry.rs:125–129` (compute_delay with AtomicU64 CAS)
- **Observation**: The jitter function uses `AtomicU64::fetch_update()` with a CAS loop to advance the PRNG state. CAS loops can spin under high contention (all retries globally share one `SEED`). A contended atomic CAS can be 50–100 ns, while an uncontended one is ~10 ns. Under heavy retry load (e.g., cascade with all tiers failing), many threads call `compute_delay()` simultaneously.
- **Hypothesized cost**: ~10–50µs total per retry (on a contended global SEED) in high-concurrency scenarios. Typically acceptable (<1% of retry latency).
- **Frequency**: warm-path-on-retries
- **Fix sketch**: (1) Use thread-local PRNG (e.g., `thread_local! { static SEED: Cell<u64> }`), so no atomic CAS needed. (2) Use a lock-free PRNG like `parking_lot::SpinMutex` if per-thread is not acceptable. (3) Measure: in production, how often do multiple threads call `compute_delay()` simultaneously? If rare, no change needed.
- **Security delta**: N/A (decorrelated jitter is preserved; only the PRNG mechanism changes)
- **Validation**: needs-bench (concurrent retry load test; measure CAS contention with perf or Criterion)

---

## Cross-cutting candidates for LazyLock/OnceLock

1. **Pricing table**: `crate::llm::pricing::model_pricing()` — convert match to `LazyLock<HashMap<&'static str, (f64, f64)>>`
2. **Cascade refusal patterns**: `crate::llm::cascade::default_refusal_patterns()` — cache in `LazyLock<Vec<String>>`
3. **Cache control JSON**: `crate::llm::anthropic::build_request_body()` — cache `serde_json::json!({"type": "ephemeral"})`
4. **Tool definitions** (if static): `crate::llm::anthropic::AnthropicProvider` — add `tools_cache: OnceLock<serde_json::Value>`
5. **Error classification patterns** (future): if regex-based, use `LazyLock<Regex>`

---

## Summary of actionable improvements

| Priority | Finding | Est. Impact | Difficulty |
|----------|---------|-------------|------------|
| High | P-LLM-1: Double serde_json pass | 500µs/call × 100 calls = 50ms/session | Medium |
| High | P-LLM-2: SSE parser String allocs | 15–25 KB alloc churn/response | High |
| High | P-LLM-16: Tool serialization per-call | 50–200µs/call with tools | Low |
| Critical | P-LLM-5: Box<dyn Future> vtable | 4–8µs per call (but I/O-dominated) | Medium |
| Critical | P-LLM-12: Cascade request cloning | 100–500 KB on escalation | Low |
| Critical | P-LLM-14: SSE data_lines join | ~1 alloc/event, compounds | High |
| Medium | P-LLM-3: Cascade gate lowercase | ~200µs on rejection | Low |
| Medium | P-LLM-4: Pricing match lookup | ~1–2µs per lookup, repeated | Low |
| Medium | P-LLM-6: Retry request cloning | 100–500 KB on transient error | Low |

### Validation summary
- **static-only**: P-LLM-3, P-LLM-4, P-LLM-7, P-LLM-11, P-LLM-17 (code inspection sufficient)
- **needs-bench**: P-LLM-1, P-LLM-2, P-LLM-5, P-LLM-6, P-LLM-10, P-LLM-16, P-LLM-18
- **code-inspection**: P-LLM-9, P-LLM-13, P-LLM-14, P-LLM-15

---

## Notes on security

All findings avoid the REJECTED categories:
- No suggestion to share a single `reqwest::Client` across tenants (F-NET-2 DNS-rebinding).
- No redirect policy relaxation (F-LLM-1 auth-header leak).
- No removal of `https_only()` or `no_proxy()` (F-NET-2).
- No skipping `SafeDnsResolver` (F-NET-2 DNS-rebinding).
- No cross-tenant token caching (F-MCP-8, F-MEM-5).
- No logging full LLM responses (F-MCP-16 PII leak).

The performance optimizations above preserve all security boundaries and are compatible with the existing threat model.
