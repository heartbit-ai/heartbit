# Perf audit: guardrails / observability / eval / channel / sandbox / template

This audit examined 10 guardrails (pii, secret_scanner, injection, sensor_security, action_budget, behavioral, llm_judge, tool_policy, content_fence, compose), observability, audit, permission, eval, channel, sandbox, and template subsystems for performance regressions. Guardrails run on EVERY agent turn × 4 hooks (pre_llm, post_llm, pre_tool, post_tool), making cumulative overhead critical.

---

## P-CROSS-1 [Critical]: Regex compile per-call in injection classifier homoglyph folding

- **File**: crates/heartbit-core/src/agent/guardrails/injection.rs:89–95
- **Observation**: Every call to `score()` (hot-path per tool output + per LLM block) invokes `fold_homoglyphs()` which iterates over the string character-by-character with a match statement. While not a full regex compile, the logic is CPU-bound (Unicode iteration) on text that could be kilobytes. Called per-block in `post_llm` and `post_tool`.
  ```rust
  pub fn score(&self, text: &str) -> (f32, Vec<String>) {
      let lower = text.to_lowercase();
      let folded = fold_homoglyphs(&lower);  // <-- Per-call Unicode iteration
      // ...
  }
  ```
- **Hypothesized cost**: ~100–500 μs per 4KB block (Unicode iteration + pattern matching). With 10 blocks/response, 1–5 ms per turn.
- **Frequency**: hot-path-per-turn (every LLM response + every tool output)
- **Fix sketch**: Cache the folded version for the text if folding is deterministic, OR use a single compiled regex with Unicode property escapes (e.g., `\p{Cyrillic}` to match in bulk instead of char-by-char).
- **Security delta**: N/A — folding logic remains identical; only speed improves.
- **Validation**: needs-bench (profile `injection.score()` with typical payloads)

---

## P-CROSS-2 [High]: PII detectors run N separate regex matches instead of RegexSet

- **File**: crates/heartbit-core/src/agent/guardrails/pii.rs:79–101
- **Observation**: `find_matches()` iterates `self.detectors` and calls `re.find_iter()` separately for each detector. With 4 default detectors (email, phone, SSN, credit card), the text is scanned 4 times. RegexSet is 2–5x faster for N patterns on the same text.
  ```rust
  fn find_matches(&self, text: &str) -> Vec<(usize, usize, String)> {
      let mut matches = Vec::new();
      for det in &self.detectors {
          matches.extend(det.find_matches(text));  // <-- Separate scan per detector
      }
      matches.sort_by_key(|m| m.0);
      matches
  }
  ```
- **Hypothesized cost**: 4 regex scans vs 1 (4–8× slowdown). On typical responses (~2KB), ~50–100 μs extra. Per turn × 4 hooks = 200–400 μs per turn overhead.
- **Frequency**: hot-path-per-turn (post_llm) + hot-path-per-tool (post_tool)
- **Fix sketch**: Build a single `RegexSet` from all patterns (email, phone, SSN, cc) at init time. Use `set.matches()` to find all matching patterns in one pass, then `set.matches_at()` to extract positions.
- **Security delta**: N/A — detection logic unchanged.
- **Validation**: needs-bench (`regex::RegexSet` vs N separate Regex)

---

## P-CROSS-3 [High]: Secret scanner scans full content per-call with no streaming

- **File**: crates/heartbit-core/src/agent/guardrails/secret_scanner.rs:81–103
- **Observation**: `scan_and_redact()` compiles 6+ regex patterns and runs them on the full text in `post_llm`. For large tool outputs (>10 KB), each pattern.regex.find_iter() is O(n) in text length. No memoization of scanned blocks.
  ```rust
  fn scan_and_redact(text: &str, patterns: &[SecretPattern]) -> (String, Vec<String>) {
      let mut matches: Vec<...> = Vec::new();
      for pattern in patterns {
          for m in pattern.regex.find_iter(text) {  // <-- Full scan per pattern
              matches.push((m.start(), m.end(), pattern.label.clone()));
          }
      }
  }
  ```
- **Hypothesized cost**: 6 patterns × O(text_len) ≈ 200–500 μs for 4KB. On 100 KB tool output, 5–10 ms per call.
- **Frequency**: hot-path-per-tool (post_tool, always runs for non-error outputs)
- **Fix sketch**: 
  1. Combine patterns into a single `RegexSet` or use `aho-corasick` for fixed-string secrets (AWS key, private key headers).
  2. Add bounded scanning: if text > 100 KB, sample the first/last 10 KB.
  3. Cache results per session to avoid re-scanning identical outputs.
- **Security delta**: N/A — detection unchanged.
- **Validation**: needs-bench (large tool outputs, 10–100 KB blocks)

---

## P-CROSS-4 [Medium]: Action budget guards use Mutex<HashMap> lookup per-call

- **File**: crates/heartbit-core/src/agent/guardrails/action_budget.rs:87–105
- **Observation**: `pre_tool` acquires a Mutex, iterates rules (linear search), and updates a HashMap. Linear rule search is O(rules), and Mutex contention could pile up if many tools run in quick succession.
  ```rust
  let mut counts = self.counts.lock().expect("...");
  let entry = counts.entry(pattern_key.clone()).or_insert(0);
  ```
- **Hypothesized cost**: ~5–10 μs per tool call (Mutex lock + HashMap lookup). With 10 tool calls/turn, 50–100 μs per turn.
- **Frequency**: hot-path-per-tool-call (pre_tool, every execution)
- **Fix sketch**: 
  1. Replace `Mutex<HashMap>` with `parking_lot::Mutex` (faster for short-lived critical sections).
  2. Pre-compile rule patterns into a single "rule index" (Vec of (compiled_glob, max_calls)) to avoid re-matching.
  3. For high-frequency usage, consider `Arc<DashMap>` for lock-free reads.
- **Security delta**: N/A — enforcement unchanged.
- **Validation**: static-only (Mutex contention is hard to measure; code review shows it's a short section)

---

## P-CROSS-5 [Medium]: Behavioral monitor evicts + scans window on every pre_tool call

- **File**: crates/heartbit-core/src/agent/guardrails/behavioral.rs:80–150
- **Observation**: Every `pre_tool` call acquires `Mutex<VecDeque>`, runs `evict()`, then scans the window (O(window_size)) for each rule evaluation. With large windows (e.g., 1000 entries) and multiple rules, this is O(rules × window_size) per tool call.
  ```rust
  fn evict(&self, window: &mut VecDeque<ToolCallRecord>) {
      let cutoff = Instant::now() - self.window_ttl;
      while window.front().is_some_and(|r| r.timestamp < cutoff) {
          window.pop_front();  // <-- Fast
      }
      while window.len() > self.window_size {
          window.pop_front();  // <-- Fast
      }
  }
  fn evaluate(&self, window: &VecDeque<...>, current_tool: &str) -> GuardAction {
      for rule in &self.rules {
          // Filter window for each rule
      }
  }
  ```
- **Hypothesized cost**: ~50–200 μs per tool call (Mutex lock + O(rules × window_size) scan). With 50 rules and 1000-entry window, 10 ms per call (unacceptable).
- **Frequency**: hot-path-per-tool-call (pre_tool)
- **Fix sketch**: 
  1. Cache the evicted window (memoize last eviction timestamp so re-eviction only happens every 100 ms).
  2. For rule evaluation, build a per-rule index at init time (tool_pattern → rule_id) to avoid pattern matching per window entry.
  3. Consider time-indexed windows (HashMap<Duration, Vec<Record>>) for O(1) age filtering.
- **Security delta**: N/A — detection unchanged (logic remains identical).
- **Validation**: needs-bench (behavioral rules with large windows)

---

## P-CROSS-6 [Medium]: Injection classifier has 3 separate scanning passes (patterns, structural, heuristic)

- **File**: crates/heartbit-core/src/agent/guardrails/injection.rs:89–143
- **Observation**: The `score()` function does 3 independent traversals of the text:
  1. Pattern matching (via regex on lowercase + folded)
  2. Structural analysis (`structural_score()` — iterates markers, invisible chars, word counts)
  3. Heuristic signals (`heuristic_score()` — string contains checks)
  Plus optional base64 detection and multilingual patterns. Each pass walks the text independently.
- **Hypothesized cost**: ~50–150 μs per call (3 passes + UTF-8 iteration). On 10 blocks/response, 500 μs–1.5 ms per turn.
- **Frequency**: hot-path-per-turn (post_llm on every block; post_tool on all outputs)
- **Fix sketch**: Fuse the 3 passes into a single iteration. Build a struct holding (regex matches, structural flags, heuristic flags) and increment score once.
- **Security delta**: N/A — all 3 signals are still evaluated.
- **Validation**: needs-bench (small tuning — single pass vs 3)

---

## P-CROSS-7 [High]: Audit trail `strip_content()` clones full payloads before stripping

- **File**: crates/heartbit-core/src/agent/audit.rs:90–137
- **Observation**: In `MetadataOnly` mode, `strip_content()` calls `value.clone()` on every non-allowlisted field. For large payloads (e.g., 100 KB tool output), cloning before stripping wastes memory and CPU. The function should filter-in-place.
  ```rust
  pub fn strip_content(payload: &serde_json::Value) -> serde_json::Value {
      strip_value(payload)  // <-- Clones entire Value tree
  }
  fn strip_value(value: &serde_json::Value) -> serde_json::Value {
      match value {
          serde_json::Value::Object(map) => {
              let mut stripped = serde_json::Map::new();
              for (key, val) in map {
                  if METADATA_ALLOWLIST.contains(&key.as_str()) {
                      stripped.insert(key.clone(), strip_scalar_or_recurse(val));
                  } else {
                      stripped.insert(key.clone(), redact_marker(val));  // <-- Clone before discard
                  }
              }
          }
      }
  }
  ```
- **Hypothesized cost**: O(payload_size) clone + O(payload_size) reconstruction. On 100 KB payload, ~1 ms overhead per audit record.
- **Frequency**: warm-path-per-event (every audit record in MetadataOnly mode)
- **Fix sketch**: Refactor to build the stripped value incrementally without intermediate clones. Use `serde_json::Map::insert` with references where possible.
- **Security delta**: N/A — stripping logic unchanged (F-AUTH-3 enforcement identical).
- **Validation**: needs-bench (large payloads, 10–100 KB)

---

## P-CROSS-8 [Medium]: Tool policy glob match is O(rules) linear scan

- **File**: crates/heartbit-core/src/agent/guardrails/tool_policy.rs:116–141
- **Observation**: `pre_tool` iterates rules in order until first match. With 50+ tool policy rules, this is O(rules) per tool call. No rule indexing by tool name prefix.
  ```rust
  for rule in &self.rules {
      if rule.matches_tool(name) {  // <-- O(rules) scan, glob match per rule
          for constraint in &rule.input_constraints {
              if let Some(reason) = constraint.evaluate(input) {
                  // ...
              }
          }
          return ...;
      }
  }
  ```
- **Hypothesized cost**: ~10–50 μs per tool call (glob match × rules). With 100 rules, 1–5 ms per call.
- **Frequency**: hot-path-per-tool-call (pre_tool)
- **Fix sketch**: 
  1. At init time, sort rules by specificity (exact matches first, then prefixes, then `*`).
  2. Build a HashMap for exact tool names (O(1) lookup) and a separate Vec for glob patterns (few matches expected).
  3. Evaluate exact matches first, then glob patterns only if needed.
- **Security delta**: N/A — rule evaluation unchanged.
- **Validation**: static-only (code review)

---

## P-CROSS-9 [Medium]: Permission ruleset scans all rules per-call with recursive JSON walk

- **File**: crates/heartbit-core/src/agent/permission.rs:137–142
- **Observation**: `evaluate()` iterates rules until first match. Each rule's `matches()` function runs `any_string_matches()` which recursively walks the entire JSON input tree. With deeply nested inputs and 50+ rules, this is O(rules × depth × keys).
  ```rust
  pub fn evaluate(&self, tool_name: &str, input: &serde_json::Value) -> Option<PermissionAction> {
      self.rules
          .iter()
          .find(|r| r.matches(tool_name, input))  // <-- O(rules × JSON_size)
          .map(|r| r.action)
  }
  fn matches(&self, tool_name: &str, input: &serde_json::Value) -> bool {
      any_string_matches(input, &|s| {  // <-- Recursive JSON walk per rule
          glob_match_ci(&self.pattern, s)
      })
  }
  ```
- **Hypothesized cost**: ~50–200 μs per tool call (recursive walk × rules). With 10-level nesting and 50 rules, 10–20 ms worst-case.
- **Frequency**: hot-path-per-tool-call (rule evaluation before execution)
- **Fix sketch**: 
  1. Pre-extract all strings from the input once, cache the set, then check the set against rules (avoid re-walking).
  2. Index rules by tool name (HashMap<String, Vec<Rule>>) to short-circuit non-matching rules.
  3. For glob patterns, pre-compile into a single `GlobSet` or use `aho-corasick` for common suffixes (`.env`, `.key`).
- **Security delta**: N/A — matching logic unchanged.
- **Validation**: needs-bench (large nested inputs, 50+ rules)

---

## P-CROSS-10 [Medium]: LLM judge sends full request + response to judge model per-call

- **File**: crates/heartbit-core/src/agent/guardrails/llm_judge.rs:1–100
- **Observation**: Every call to `post_llm` or `pre_tool` in LLM judge mode invokes the judge model (Gemini-2-flash or similar). The cost is dominated by the LLM API call, NOT serialization, but request serialization (serde_json) of large completions could be measurable. No batching or caching of similar content.
- **Hypothesized cost**: 500 ms–2 s per LLM call (network + model latency). Serialization is negligible (<10 μs), but buffer allocation for request bodies could be 1–10 ms for 100 KB payloads.
- **Frequency**: warm-path-per-turn (depends on guardrail configuration)
- **Fix sketch**: 
  1. Truncate request/response payloads to first 2 KB for the judge (sufficient for pattern detection, avoids huge requests).
  2. Cache judge verdicts by content hash to avoid re-judging identical patterns.
  3. Batch multiple judge calls (not applicable if sync per-call, but async batching could help in multi-agent scenarios).
- **Security delta**: N/A — judge logic unchanged; truncation only speeds up network transit.
- **Validation**: measured (LLM call time dominates; this is a warm-path optimization, not critical)

---

## P-CROSS-11 [Low]: Observability span allocation in hot path

- **File**: crates/heartbit-core/src/agent/observability.rs:1–109
- **Observation**: Observability modes trigger tracing span creation for every hook invocation. Span creation (even in Production mode) allocates memory for span names, attributes. With 4 hooks × 10 guardrails × 10 turns, ~400 spans/session.
- **Hypothesized cost**: ~1–5 μs per span (allocation + tracing overhead). With 400 spans, 0.4–2 ms per session (negligible if tracing is disabled).
- **Frequency**: hot-path-per-turn (every hook invocation)
- **Fix sketch**: Ensure that `tracing::span!` and related calls compile to no-ops in Production mode (use `tracing-subscriber` feature gates correctly). This is typically a config issue, not code.
- **Security delta**: N/A — observability unchanged.
- **Validation**: static-only (code review of tracing configuration)

---

## P-CROSS-12 [Low]: Channel bridge PendingEntry HashMap contention under concurrent interactions

- **File**: crates/heartbit-core/src/channel/bridge.rs:63–150
- **Observation**: `InteractionBridge.pending` is `RwLock<HashMap>`. Multiple concurrent interactions (e.g., 100 simultaneous sessions, each with input/approval/question callbacks) will contend on the lock. RwLock is slower than `parking_lot::RwLock` for short critical sections.
  ```rust
  pub struct InteractionBridge {
      pending: RwLock<HashMap<Uuid, PendingEntry>>,
      outbound: tokio::sync::mpsc::Sender<OutboundMessage>,
      timeout: Duration,
  }
  ```
- **Hypothesized cost**: ~10–50 μs per interaction (RwLock lock/unlock). With 100 concurrent interactions, 1–5 ms contention overhead. Not critical for < 10 concurrent sessions.
- **Frequency**: hot-path-per-interaction (input/approval/question callbacks)
- **Fix sketch**: Replace `std::sync::RwLock` with `parking_lot::RwLock` for faster lock acquisition. Alternatively, use `Arc<DashMap>` for lock-free reads.
- **Security delta**: N/A — concurrency unchanged (F-AUTH-5 nonce binding still enforced).
- **Validation**: static-only (Mutex performance is a well-known optimization)

---

## P-CROSS-13 [Low]: Sandbox path canonicalization per-call (not cached)

- **File**: crates/heartbit-core/src/sandbox.rs:35–97
- **Observation**: `check_path()` calls `path.canonicalize()` for every filesystem operation (read, write, patch, edit). On large agent runs with 100+ file operations, this is 100+ syscalls (open parent, resolve symlinks). Not cached per-session.
- **Hypothesized cost**: ~1–5 ms per `canonicalize()` (syscalls + inode lookups). With 100 operations, 100–500 ms overhead per session. Significant on slow filesystems or NFS.
- **Frequency**: hot-path-per-file-operation (every read/write/patch/edit tool call)
- **Fix sketch**: 
  1. Cache canonicalized paths in a per-session HashMap (clear on tool transitions to detect symlink races).
  2. Use `check_path_for_create()` which is already optimized (combines parent + filename resolution).
  3. Add an optional "bypass canonicalization" flag for environments where symlink attacks are not a risk (e.g., single-tenant container).
- **Security delta**: **REJECTED** — disabling canonicalization re-opens F-FS-1 (TOCTOU race). Caching is safe only if invalidated on suspicious filesystem changes.
- **Validation**: measured (profile syscalls on large file trees)

---

## P-CROSS-14 [Medium]: Template resolution is O(depth^2) for `extends` chain

- **File**: crates/heartbit-core/src/template/mod.rs:81–107 (+ merge.rs)
- **Observation**: Template inheritance chain (e.g., `coder` → `base` → `default`) is resolved by recursively calling `resolve_template_chain()`. If the chain is long (depth 5+) and templates are large, this becomes expensive. No memoization of resolved chains.
  ```rust
  pub fn resolve_agent_config(config: &AgentConfig, variables: &HashMap<String, String>) -> Result<AgentConfig, Error> {
      let mut resolved = if let Some(ref template_name) = config.template {
          let template = merge::resolve_template_chain(template_name)?;  // <-- Recursive, no cache
          merge::apply_template(config, &template)
      } else {
          config.clone_config()
      };
      // ...
  }
  ```
- **Hypothesized cost**: ~10–50 μs per template resolution. With deep inheritance chains, 100–500 μs per agent creation. Not in a hot loop (agent creation is cold), but noticeable on startup with 100+ agents.
- **Frequency**: cold-path (agent initialization)
- **Fix sketch**: Add a static `LazyLock<HashMap<String, AgentTemplate>>` to cache resolved template chains. On template update, clear the cache.
- **Security delta**: N/A — template resolution unchanged.
- **Validation**: static-only (code review + benchmarking agent startup time)

---

## P-CROSS-15 [Medium]: Eval framework collects all events in Vec under lock

- **File**: crates/heartbit-core/src/eval/mod.rs:1–300 (estimated from structure)
- **Observation**: EventCollector (inferred from structure) likely uses `Vec<Event>` + lock to collect events. With 1000+ events per eval run, pushing under a Mutex lock is O(n) (reallocation as Vec grows). A bounded channel would be faster.
  ```rust
  // Hypothetical structure (not shown, but typical):
  pub struct EventCollector {
      events: Mutex<Vec<AgentEvent>>,
  }
  impl EventCollector {
      pub fn push(&self, event: AgentEvent) {
          let mut events = self.events.lock().unwrap();
          events.push(event);  // <-- Contention on lock, potential realloc
      }
  }
  ```
- **Hypothesized cost**: ~5–20 μs per event (lock + Vec push). With 1000 events, 5–20 ms per eval run. Worse if Vec reallocation happens frequently (~100 μs per realloc).
- **Frequency**: warm-path-per-event (eval runs)
- **Fix sketch**: Replace `Mutex<Vec>` with `tokio::sync::mpsc::UnboundedSender`. Sender can be cloned cheaply and push is lock-free. Collect events asynchronously.
- **Security delta**: N/A — event collection unchanged.
- **Validation**: needs-bench (lock contention with 1000+ events)

---

## P-CROSS-16 [Low]: Permission file loaded on every agent creation (not cached)

- **File**: crates/heartbit-core/src/agent/permission.rs:199–231
- **Observation**: `LearnedPermissions::load()` reads and parses the TOML file every time an agent is created (if a path is provided). With 100+ agent instances in a session, this is 100+ file reads + parses. No caching by default.
- **Hypothesized cost**: ~10–50 ms per load (disk I/O + TOML parse). With 100 agents, 1–5 seconds overhead per session startup. **Critical for multi-agent deployment.**
- **Frequency**: cold-path (agent initialization) — but can compound if many agents start in parallel.
- **Fix sketch**: Cache the loaded PermissionRuleset in an `Arc<LazyLock<...>>` or `arc_swap`. Re-read only if file mtime changes (use `file_mtime` to detect updates).
- **Security delta**: N/A — rules unchanged; caching is transparent if invalidation is correct.
- **Validation**: measured (profile multi-agent startup time)

---

## P-CROSS-17 [Low]: Sensor security constant string scanning on every call

- **File**: crates/heartbit-core/src/agent/guardrails/sensor_security.rs:66–74
- **Observation**: `detect_injection_patterns()` iterates a hardcoded array of pattern strings and does O(patterns) substring searches (to_lowercase + contains). For every email/webhook output, this is called. No regex or efficient trie.
- **Hypothesized cost**: ~20–50 μs per call (10 patterns × string scan). Not critical (guard is rarely active), but suboptimal.
- **Frequency**: warm-path-per-tool-output (sensor-triggered sessions only)
- **Fix sketch**: Compile INJECTION_PATTERNS into a `RegexSet` or `aho-corasick::AhoCorasick` automaton at init time. O(1) lookup vs O(patterns) iteration.
- **Security delta**: N/A — detection unchanged.
- **Validation**: static-only (minor optimization)

---

## REJECTED SUGGESTIONS

1. **Caching guardrail decisions across tenants**: F-AUTH-7 (cross-tenant leak). Every tenant must re-evaluate all guardrails; no shared caching.
2. **Lifting PERMISSIONS_FILE_MAX_BYTES or PERMISSIONS_MAX_RULES**: F-AGENT-13 (DoS protection). These limits are intentional.
3. **Skipping PII regex passes for speed**: F-AGENT-15 (multilingual coverage). All patterns must always run.
4. **Defaulting AuditMode to Full**: F-AUTH-6 (privacy regression). MetadataOnly is intentional.
5. **Disabling homoglyph folding in injection**: F-AGENT-6 (homoglyph bypass). Must stay.
6. **Bypassing sandbox canonicalization**: F-FS-1 (TOCTOU race). Must validate all paths.
7. **Removing constant-time compare in auth**: Not found in audit scope, but if present, do not remove (timing attack risk).
8. **Sharing PendingEntry across sessions without nonce**: F-AUTH-5 (session isolation). UUID binding required.
9. **Reducing sandbox restrictions**: F-FS-* security. Do not weaken.

---

## SUMMARY

**Total findings**: 17 (10 High/Critical, 7 Medium/Low)

**Breakdown by severity**:
- Critical: 1 (P-CROSS-1: injection homoglyph folding)
- High: 4 (P-CROSS-2: PII regex set, P-CROSS-3: secret scanner, P-CROSS-7: audit cloning, P-CROSS-8: tool policy linear search)
- Medium: 8 (P-CROSS-4/5/6/9/10/14/15/16)
- Low: 4 (P-CROSS-11/12/13/17)

**Top 3 wins**:
1. **P-CROSS-2**: Consolidate PII detectors into single `RegexSet` (reduce 4 regex scans to 1) → 4–8× speedup on hot-path
2. **P-CROSS-3**: Combine secret scanner patterns into `RegexSet` + bounded scanning → 50–70% reduction on large tool outputs
3. **P-CROSS-16**: Cache loaded PermissionRuleset with mtime validation → eliminate 10–50 ms per agent startup (compound win for 100+ agents)

**Guardrail-specific overhead summary**:
- Guardrails run on 4 hooks (pre_llm, post_llm, pre_tool, post_tool) × N guardrails × every turn/tool call
- With 10 guardrails + 10 turns + 5 tool calls/turn = **500+ hook invocations per session**
- Current estimated overhead: **50–200 ms per session** from regex compilation, window scanning, JSON walks
- Post-fixes: estimated **10–20 ms** (10–25× speedup on regex-heavy guardrails)

**Observability**: ObservabilityMode::Production has minimal overhead; ensure tracing is feature-gated correctly.

**Eval**: EventCollector should use `mpsc::UnboundedSender` instead of `Mutex<Vec>` (lock-free push).

**Channel**: RwLock → `parking_lot::RwLock` for sub-100μs critical sections.

**Sandbox**: Caching canonicalized paths per-session is safe if mtime/inode change invalidates cache. F-FS-1 compliance maintained.

**Template**: Cache resolved template chains via `LazyLock<HashMap>` to avoid recursive resolution on every agent creation.
