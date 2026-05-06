# Perf audit: agent loop

Audit date: 2026-05-06 | Scope: `runner.rs` (2355 lines), `mod.rs`, `orchestrator.rs`, `context.rs`, `pruner.rs`, `cache.rs`, `events.rs`

---

## P-RUNNER-1 [Critical]: `tool_defs.clone()` per execute → O(N) allocation on every turn

- **File**: `crates/heartbit-core/src/agent/runner.rs:374, 397`
- **Observation**:
  ```rust
  pub async fn execute(&self, task: &str) -> Result<AgentOutput, Error> {
      let ctx = AgentContext::new(&self.system_prompt, task, self.tool_defs.clone())
  }
  pub async fn execute_with_content(...) -> Result<AgentOutput, Error> {
      let ctx = AgentContext::from_content(&self.system_prompt, content, self.tool_defs.clone())
  }
  ```
  Every invocation of `execute()` or `execute_with_content()` clones the entire `Vec<ToolDefinition>`. Each `ToolDefinition` contains `name: String`, `description: String`, `input_schema: JSONSchema`, and potentially large nested schema objects.
- **Hypothesized cost**: Per-call ~1–5 ms (O(N) string copies, 50–500 ToolDefinitions typical). Cumulative: ~5% of LLM-bounded workloads where network dominates, but on GPU-provider latency or batch scenarios with <50ms think time, this becomes visible.
- **Frequency**: Once per session (hot-path-per-execute, not per-turn), but execute is the public API entry point; estimated 1–100 sessions/min in production.
- **Fix sketch**: Wrap `tool_defs` in `Arc<Vec<ToolDefinition>>` at construction (already done for `tools: Arc<dyn Tool>`). Pass `Arc` to `AgentContext::new()` instead of `Vec`, and store `Arc` in context. Dereference on demand in `to_request()`.
- **Security delta**: N/A
- **Validation**: needs-bench (measure `clone()` cost with realistic tool counts; estimate: 20–100 µs × 50 tools = 1–5 ms)

---

## P-RUNNER-2 [High]: `TokenUsage` Copy→Add overhead per turn

- **File**: `crates/heartbit-core/src/agent/runner.rs:809, 1949, 1681, 1583, etc.` (~20+ sites use `+=`)
- **Observation**:
  ```rust
  pub struct TokenUsage { pub input_tokens: u32, pub output_tokens: u32, ...}
  // impl not shown, but derive(Copy, Clone) should be confirmed
  total_usage += response.usage;  // line 809
  *usage_acc += resp.usage;  // line 1949
  total_usage += summary_usage;  // line 1583
  ```
  If `TokenUsage` does **not** derive `Copy`, each `+=` triggers an implicit clone. Across ~50 turns + recursive summarization + tool output compression, this adds up.
- **Hypothesized cost**: If `Copy` is missing: ~10–20 µs per `+=` × 50 assignments/turn × 10 turns = 5–10 ms per run. If `Copy` present: ~0 (already optimized).
- **Frequency**: hot-path-per-turn (multiple +=per turn in normal case, more in compression/summarization)
- **Fix sketch**: Verify `TokenUsage` derives `Copy` (or inline it). If not, add derive or use `std::mem::replace()` instead of `+=` for non-Copy case (though Copy is cleaner).
- **Security delta**: N/A
- **Validation**: static-only (grep for `derive` in types.rs; if Copy present, finding is invalid)

---

## P-RUNNER-3 [High]: Redundant `recently_used_tools` clone per turn

- **File**: `crates/heartbit-core/src/agent/runner.rs:1510`
- **Observation**:
  ```rust
  recently_used_tools = allowed_calls.iter().map(|c| c.name.clone()).collect();
  ```
  Every turn that executes tools, a new `Vec<String>` is allocated and populated. For 30 turns × 5 tools, that's ~30 allocations of ~40 bytes + string copies. In the next turn, this vector is passed to `select_tools_for_turn()` where it's searched via `.contains(&tool.name)` (O(N) scan per tool).
- **Hypothesized cost**: ~100–200 µs per turn (allocate + copy 5–10 strings). Across 30 turns = 3–6 ms. Secondary: O(N²) tool-candidate scanning in `select_tools_for_turn` (line 1896: `recently_used.contains()` called per-tool, not per-turn).
- **Frequency**: hot-path-per-turn (conditional on tools being executed)
- **Fix sketch**: (1) Use `HashSet<&str>` (single allocation, O(1) contains). (2) Pre-allocate with `Vec::with_capacity(allowed_calls.len())` if Vec must be used. (3) For `select_tools_for_turn`, convert to HashSet upfront.
- **Security delta**: N/A
- **Validation**: needs-bench (measure Vec::contains vs HashSet in tool count 20–100 range)

---

## P-RUNNER-4 [High]: `DoomLoopTracker::hash_tool_calls()` hashes JSON strings per turn

- **File**: `crates/heartbit-core/src/agent/doom_loop.rs:41–53`
- **Observation**:
  ```rust
  fn hash_tool_calls(calls: &[ToolCall]) -> u64 {
      let mut sorted: Vec<(String, String)> = calls
          .iter()
          .map(|tc| (tc.name.clone(), tc.input.to_string()))  // ← to_string() on serde_json::Value
          .collect();
      sorted.sort();
      let mut hasher = DefaultHasher::new();
      for (name, input) in &sorted {
          name.hash(&mut hasher);
          input.hash(&mut hasher);  // hashing full JSON string
      }
      hasher.finish()
  }
  ```
  Per-turn (line 1508–1518 in runner.rs), this hashes the entire JSON input of every tool call. For a 10KB JSON input, this means serializing to string + hashing the full string. With 10 tools, 5 turns: ~100 allocs + serial string hashing.
- **Hypothesized cost**: ~100–500 µs per turn (depends on tool input size, typically 1–10KB per call). Across 10 turns = 1–5 ms.
- **Frequency**: hot-path-per-turn (called via `doom_tracker.record()` at line 1508, every iteration)
- **Fix sketch**: (1) Use a **hash of the serde_json::Value** directly (e.g., `serde_json::to_value` then hash the serialized bytes once). (2) Cache the hash in `ToolCall` if the LLM provider doesn't already compute it. (3) Use `fnv1a_hash` (faster than DefaultHasher for small strings).
- **Security delta**: N/A (hashing is idempotent; changing algorithm doesn't affect doom-loop detection logic)
- **Validation**: needs-bench (measure to_string() + sort + hash for realistic tool counts/sizes)

---

## P-RUNNER-5 [High]: Cache key computation via `.split()` + `.to_lowercase()` per turn

- **File**: `crates/heartbit-core/src/agent/runner.rs:1872–1888 (select_tools_for_turn)`
- **Observation**:
  ```rust
  let recent_text: String = messages
      .iter()
      .rev()
      .take(4)
      .flat_map(|m| m.content.iter())
      .filter_map(|block| match block { ... })
      .collect::<Vec<_>>()
      .join(" ")
      .to_lowercase();
  
  let keywords: Vec<&str> = recent_text
      .split(|c: char| !c.is_alphanumeric() && c != '_')
      .filter(|w| w.len() > 2)
      .collect();
  
  for tool in all_tools {
      let tool_text = format!("{} {}", tool.name, tool.description).to_lowercase();
      let score = keywords.iter().filter(|kw| tool_text.contains(**kw)).count();
  }
  ```
  Every turn with dynamic tool selection (line 584): (1) collect 4 messages into Vec, join into String, to_lowercase(); (2) split by char predicate (O(N) regex-like operation); (3) for each tool, allocate new String via format!, to_lowercase(), then linear search for each keyword. For 100 tools + 10 keywords, this is ~1000 contains checks.
- **Hypothesized cost**: ~1–5 ms per turn (depends on message size, typically 5–50KB context). Overhead is significant only when `max_tools_per_turn` is set <40 (otherwise short-circuit at line 1868).
- **Frequency**: warm-path-per-turn (only when `max_tools_per_turn` is configured)
- **Fix sketch**: (1) Cache keywords in turn state (don't recompute per tool). (2) Use regex (once-compiled, stored in lazy_static) instead of char predicate. (3) For scoring, pre-lowercase tool descriptions at construction time (store in tool registry). (4) Use `contains(&word)` with a pre-compiled set of keywords.
- **Security delta**: N/A
- **Validation**: needs-bench (compare current split+lowercase loop vs regex+pre-computed keywords)

---

## P-RUNNER-6 [High]: `find_closest_tool()` called per unknown tool, hashes full tool name map

- **File**: `crates/heartbit-core/src/agent/runner.rs:869–887, 1966–1973`
- **Observation**:
  ```rust
  for call in tool_calls.iter_mut() {
      if !self.tools.contains_key(&call.name)
          && let Some(repaired) = self.find_closest_tool(&call.name, 2)
      {
          // repair...
      }
  }
  
  pub(super) fn find_closest_tool(&self, name: &str, max_distance: usize) -> Option<&str> {
      self.tools
          .keys()
          .map(|k| (k.as_str(), levenshtein(name, k)))  // ← O(M*L) per key
          .filter(|(_, d)| *d <= max_distance && *d > 0)
          .min_by_key(|(_, d)| *d)
          .map(|(name, _)| name)
  }
  ```
  Each unknown tool name triggers a full scan of the tool registry (HashMap<String, Arc<dyn Tool>>) with Levenshtein distance O(M×L) where M = tool name length, L = unknown name length. For 100 tools + unknown name of length 20, this is ~2000 ops per unknown call. Worst case: 10 unknown tools per turn = 20K ops.
- **Hypothesized cost**: ~100–500 µs per unknown tool call (Levenshtein on 100 tools). With retry logic in permission/guardrail loops, can be called 2–5× per "bad" turn = 500 µs–2.5 ms.
- **Frequency**: cold-path in normal operation, but hot under adversarial or jailbreak scenarios (attackers probe with misspelled tool names to trigger repairs)
- **Fix sketch**: (1) Precompute a **trie or BK-tree of tool names** at construction time; Levenshtein queries drop to O(log N + distance²). (2) Cache recent lookups (LRU: last 10 unknown names). (3) Limit max_distance to 1 (most typos are single-char) — already capped at 2.
- **Security delta**: **CRITICAL: DO NOT remove or modify this check** (F-AGENT-1, security audit). This audit event is essential for detecting Levenshtein bypass attempts. However, the implementation can be optimized without changing semantics.
- **Validation**: static-only (code review of Levenshtein cost and call frequency)

---

## P-RUNNER-7 [High]: `execute_tools_parallel()` call_ids/call_names cloned per tool

- **File**: `crates/heartbit-core/src/agent/runner.rs:2029–2041`
- **Observation**:
  ```rust
  let call_ids: Vec<String> = calls.iter().map(|c| c.id.clone()).collect();
  let call_names: Vec<String> = calls.iter().map(|c| c.name.clone()).collect();
  let mut join_set = tokio::task::JoinSet::new();
  
  for (idx, call) in calls.iter().enumerate() {
      let tool = self.tools.get(&call.name).cloned();
      let input = call.input.clone();
      let call_name = call.name.clone();
      let timeout = self.tool_timeout;
      
      self.emit(AgentEvent::ToolCallStarted {
          agent: self.name.clone(),
          tool_name: call.name.clone(),
          tool_call_id: call.id.clone(),
          ...
      });
  }
  ```
  Three unnecessary allocations per tool:
  1. `call_ids` and `call_names` vecs (unused — only `call.id` and `call.name` used in loop)
  2. `call.name.clone()` again inside the loop (line 2041: `let call_name = call.name.clone()`)
  3. `tool.cloned()` (line 2039) — clones Arc contents (cheap) but Arc itself should be moved, not cloned

  For 10 tools per turn × 10 turns = 100 clones of String + 100 redundant "unused vecs" = ~10K allocations.
- **Hypothesized cost**: ~10–20 µs per tool (allocate + clone string). Across 10 tools = 100–200 µs per turn. Across 30 turns = 3–6 ms.
- **Frequency**: hot-path-per-tool (called in `execute_tools_parallel`, which is called once per turn with tools)
- **Fix sketch**: (1) Remove lines 2029–2030 (unused). (2) Use `&str` or borrow in loop instead of cloning. (3) For captured variables in spawn closure, use `.to_string()` only when needed (e.g., for emit), and defer cloning to async block.
- **Security delta**: N/A
- **Validation**: static-only (code review)

---

## P-RUNNER-8 [High]: `resp.clone()` caching CompletionResponse per cache hit

- **File**: `crates/heartbit-core/src/agent/runner.rs:689`
- **Observation**:
  ```rust
  if let (Ok(resp), Some(key)) = (&result, cache_key)
      && resp.stop_reason == crate::llm::types::StopReason::EndTurn
      && let Some(ref c) = self.response_cache
  {
      c.put(key, resp.clone());  // ← Clone full CompletionResponse
  }
  ```
  `CompletionResponse` contains `Vec<ContentBlock>`, and each `ContentBlock` is cloned. For a typical response (2KB text + metadata), this is ~2KB allocation + copy. Across 30 turns with cache, if 50% hit rate: ~15 cache puts = 30KB allocation + copy overhead. Secondary issue: in `ResponseCache::get()` (cache.rs:45), the entry is cloned again on retrieval.
- **Hypothesized cost**: ~100–500 µs per cache put/get (depends on response size, typically 2KB). Across 15 cache operations = 1.5–7.5 ms per run.
- **Frequency**: warm-path (conditional on response_cache being enabled and cache hit)
- **Fix sketch**: (1) Store `Arc<CompletionResponse>` in cache instead of owned response — avoids clones on put/get. (2) Use `Arc` in ResponseCache definition. (3) Update compute_key to work with Arc.
- **Security delta**: N/A (Arc is transparent to caching logic)
- **Validation**: needs-bench (measure clone overhead for typical CompletionResponse sizes)

---

## P-RUNNER-9 [Medium]: Recursive summarization allocates per cluster

- **File**: `crates/heartbit-core/src/agent/runner.rs:1673–1707`
- **Observation**:
  ```rust
  let mut cluster_summaries = Vec::new();
  
  for chunk in lines.chunks(cluster_size) {
      let cluster_text = chunk.join("\n");  // ← allocate per chunk
      let (summary, usage) = self.summarize_text(&cluster_text).await?;
      total_usage += usage;
      match summary {
          Some(s) => cluster_summaries.push(s),
          ...
      }
  }
  
  let combined = format!(
      "Summarize the following section summaries into one cohesive summary:\n\n{}",
      cluster_summaries
          .iter()
          .enumerate()
          .map(|(i, s)| format!("Section {}:\n{}", i + 1, s))  // ← alloc per section
          .collect::<Vec<_>>()
          .join("\n\n")
  );
  ```
  Multiple unnecessary allocations: (1) `.join("\n")` per chunk; (2) `collect::<Vec<_>>()` of formatted strings; (3) final `.join()`. For a 100-line conversation split into 10 chunks, this is 10 joins + 10 formats + 1 join = ~20 allocations.
- **Hypothesized cost**: ~1–2 ms (dominated by LLM calls, not allocation, but avoidable overhead)
- **Frequency**: cold-path (only when context exceeds threshold, typically once per long run)
- **Fix sketch**: (1) Use single String buffer with `write!()` instead of format! chains. (2) Pre-allocate cluster_summaries with capacity. (3) Avoid intermediate `Vec<_>` collect.
- **Security delta**: N/A
- **Validation**: static-only (code review of allocation patterns)

---

## P-RUNNER-10 [Medium]: Permission rules accessed via RwLock every pre_tool check

- **File**: `crates/heartbit-core/src/agent/runner.rs:1412–1413, 280–289`
- **Observation**:
  ```rust
  fn eval_permission(
      &self,
      tool_name: &str,
      input: &serde_json::Value,
  ) -> Option<permission::PermissionAction> {
      self.permission_rules
          .read()  // ← RwLock::read() per tool
          .expect("permission rules lock poisoned")
          .evaluate(tool_name, input)
  }
  ```
  Every tool call (per-tool in loop at line 1410–1470) acquires a read lock on `permission_rules`. With 10 tools per turn × 30 turns = 300 lock acquisitions. While RwLock read is cheap (~10–50 ns), repeated lock/unlock cycles waste instruction cache and can suffer contention if permission evaluation is slow (O(N) rule evaluation inside the lock).
- **Hypothesized cost**: ~100–500 ns per lock × 300 = 30–150 µs per run (negligible in isolation, but contributes to 0.1–1% overhead).
- **Frequency**: hot-path-per-tool
- **Fix sketch**: (1) Consider `parking_lot::RwLock` (faster, optimized for read-heavy) if not already in use. (2) Snapshot permission rules at turn start (if mutable state is not needed mid-turn). (3) Use atomic flag for common case (no rules).
- **Security delta**: N/A (locking semantics unchanged)
- **Validation**: static-only (code review; benchmark only if profiling shows >1% overhead)

---

## P-RUNNER-11 [Medium]: `OnEvent` callback cloned per emit

- **File**: `crates/heartbit-core/src/agent/runner.rs:299–303`
- **Observation**:
  ```rust
  fn emit(&self, event: AgentEvent) {
      if let Some(ref cb) = self.on_event {
          cb(event);  // ← Arc<dyn Fn> is cloned inside emit call (implicit)
      }
  }
  ```
  Each emit event (20+ per turn: TurnStarted, LlmResponse, ToolCallStarted, ToolCallCompleted, GuardrailWarn, etc.) calls `cb(event)`. The callback is stored as `Arc<dyn Fn>`, so passing to the callback involves Arc refcount update if the callback signature expects ownership. Typically ~10 ns × 50 events = 500 ns overhead per turn. Not a hot issue, but worth noting.
- **Hypothesized cost**: ~10 ns per Arc clone × 50 events = 500 ns per turn. Negligible.
- **Frequency**: hot-path-per-turn (every event emission)
- **Fix sketch**: No change needed (Arc::clone is O(1) and cheap). This is an idiomatic pattern.
- **Security delta**: N/A
- **Validation**: static-only (Arc operations are constant-time)

---

## P-RUNNER-12 [Medium]: `Message` cloned in guardrail loops

- **File**: `crates/heartbit-core/src/agent/runner.rs:1002–1010, 1047–1050`
- **Observation**:
  ```rust
  ctx.add_assistant_message(Message {
      role: crate::llm::types::Role::Assistant,
      content: vec![ContentBlock::Text { ... }],
  });
  
  // Later, response.content is moved:
  ctx.add_assistant_message(Message {
      role: crate::llm::types::Role::Assistant,
      content: response.content,  // ← move, good
  });
  ```
  Mixed patterns: sometimes Message is constructed inline (no clone), sometimes content is passed (good). But in guardrail denial flow (line 1002–1006), a new Message is created with trivial content. This is fine. However, earlier in the turn (line 1047–1050), `response.content` is moved directly. **No issue found here—code is already optimized.**
- **Hypothesized cost**: Already optimized; 0 µs overhead.
- **Frequency**: hot-path-per-turn
- **Fix sketch**: N/A (already best-practice)
- **Security delta**: N/A
- **Validation**: static-only

---

## P-RUNNER-13 [Low]: `name.to_string()` and `e.to_string()` repeated in event emission

- **File**: `crates/heartbit-core/src/agent/runner.rs:506–507, 527, 565, 595, 802, etc.` (~30+ sites)
- **Observation**:
  ```rust
  self.emit(AgentEvent::RunStarted {
      agent: self.name.clone(),
      task: task.to_string(),  // ← to_string() on &str
  });
  
  self.emit(AgentEvent::RunFailed {
      agent: self.name.clone(),
      error: e.to_string(),  // ← to_string() on Error enum
      ...
  });
  ```
  Every event includes `self.name.clone()` (should be `self.name.as_str()` if AgentEvent fields allow), and every error emission includes `e.to_string()` (allocates new String). Across 50+ events per run, this is ~50 string allocations. For a 20-char name × 50 events = 1KB overhead.
- **Hypothesized cost**: ~5–10 µs per event (allocate + copy string) × 50 = 250–500 µs per run. Negligible but eliminable.
- **Frequency**: hot-path-per-turn (event emission)
- **Fix sketch**: (1) Check if AgentEvent fields can be `&str` (borrowed) instead of `String` (owned). If events are serialized (serde), ownership may be required. (2) If ownership is needed, accept it but avoid `.clone()` on self.name—use arc::str or pre-allocate events with Cow.
- **Security delta**: N/A
- **Validation**: static-only (code review of event field types and serialization)

---

## P-RUNNER-14 [Low]: Unnecessary `Vec::new()` followed by `.push()`

- **File**: `crates/heartbit-core/src/agent/runner.rs:517, 1408, 1675, etc.`
- **Observation**:
  ```rust
  let mut recently_used_tools: Vec<String> = Vec::new();  // line 517
  // used later, but capacity is unknown
  
  let mut allowed = Vec::new();  // line 1408
  let mut denied = Vec::new();
  for call in tool_calls {
      // ... add to allowed or denied
  }
  ```
  Many `Vec::new()` are followed by pushes without pre-capacity. For vectors with known size (e.g., `allowed_calls.len()`), use `Vec::with_capacity()`.
- **Hypothesized cost**: ~10 µs per Vec (allocation + realloc), negligible if reallocations are rare. But a few sites (e.g., line 1408 processing all tool calls) should use `with_capacity`.
- **Frequency**: warm-path
- **Fix sketch**: (1) Replace `Vec::new()` with `Vec::with_capacity(expected_size)` at 5–10 sites. (2) Low ROI; focus on higher-impact findings first.
- **Security delta**: N/A
- **Validation**: static-only

---

## P-RUNNER-15 [Low]: `.to_string()` in error messages / placeholders

- **File**: `crates/heartbit-core/src/agent/runner.rs:1751, 1954, 2121`
- **Observation**:
  ```rust
  content.clone()  // line 1751
  format!("{compressed}\n[compressed from {original_len} bytes]")  // line 1954
  ToolOutput::error(e.to_string())  // line 2122
  ```
  Multiple sites use `.to_string()` or `.clone()` for temporary error messages or formatting. These are typically cold-path (errors, compression) but still add up.
- **Hypothesized cost**: ~10–20 µs per error path. Across 30 turns = ~300 µs if errors occur. Typically errors are rare, so negligible.
- **Frequency**: cold-path
- **Fix sketch**: Use `Cow<str>` or `&str` references where possible; avoid allocating for error strings.
- **Security delta**: N/A
- **Validation**: static-only

---

## REJECTED SUGGESTIONS

1. **Caching results across tenants to improve hit rate (F-MEM-5, F-AUTH-5 re-open)**
   - **Reason**: Security audit explicitly flags cross-tenant cache contamination (F-AGENT-3: cache key without tenant_id can leak). The current code uses `compute_key_scoped()` to mitigate. Extending cache TTL or deduplication would re-expose this vulnerability.

2. **Removing tool-name repair audit event (F-AGENT-1)**
   - **Reason**: Security audit identifies Levenshtein-based bypass of permission rules as Critical. The `ToolNameRepaired` event is the *only* audit trail. Removing it masks attacks. Instead, optimize the Levenshtein search (BK-tree, trie, caching) without removing the event.

3. **Sharing one `reqwest::Client` across `IpPolicy` contexts (F-NET-2)**
   - **Reason**: Security audit flags DNS rebinding as a bypassing vector for blocklist checks. Each policy context must use isolated DNS resolution. Sharing would collapse per-policy DNS behavior.

4. **Removing nonce-bearing cwd marker (F-FS-8)**
   - **Reason**: Security audit flags `__HEARTBIT_CWD__` injection as a hijacking vector. Removing it would allow stdout cwd spoofing. Instead, use process-scoped unique prefix (e.g., `__HEARTBIT_<PID>_CWD__`).

5. **Replacing `subtle::ConstantTimeEq` with `==` on tokens**
   - **Reason**: Security audit implicitly requires constant-time token comparison to prevent timing-based token theft. Using `==` re-opens timing attacks.

---

## Summary

**Total findings**: 15 (including REJECTED)  
**Breakdown by severity**:
- **Critical**: 1 (P-RUNNER-1: tool_defs clone, 5–10% latency on per-call basis)
- **High**: 7 (P-RUNNER-2 through P-RUNNER-8: TokenUsage, recently_used, doom hash, keywords, find_closest_tool, tool ID clones, cache resp clone)
- **Medium**: 4 (P-RUNNER-9 through P-RUNNER-12: recursive summarization, RwLock, OnEvent, Message)
- **Low**: 3 (P-RUNNER-13 through P-RUNNER-15: name.to_string, Vec::new, error formatting)

**Top 3 quick wins** (feasible, high impact):
1. **P-RUNNER-1**: Wrap `tool_defs` in Arc (1–5 ms saving per session start).
2. **P-RUNNER-3**: Use HashSet for recently_used_tools (100–200 µs per turn → 3–6 ms per run with dynamic tool selection).
3. **P-RUNNER-5**: Pre-compute lowercase tool descriptions at build time + use regex (1–5 ms per turn with max_tools_per_turn).

**Validation strategy**:
- Static review: 10 findings (no benchmark needed, obvious optimizations).
- Benchmark required: 5 findings (P-RUNNER-1, P-RUNNER-2, P-RUNNER-4, P-RUNNER-5, P-RUNNER-8).
- No action: 2 findings (already optimal: P-RUNNER-11, P-RUNNER-12).

**Security-aware approach**: All optimizations preserve audit invariants (caching is scoped, Levenshtein repair is audited, permissions remain enforced). No findings conflict with `security-audit-heartbit-core-2026-05-06.md`.
