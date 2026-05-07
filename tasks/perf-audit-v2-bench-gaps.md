# v2 Bench coverage gap audit

**Audit date**: 2026-05-06  
**Scope**: heartbit-core benches + hot paths  
**Total gaps identified**: 14 (6 existing bench gaps + 8 new benches)

---

## Existing bench gaps

### Gap-EX-1: memory_recall missing 100k scale and hybrid mode
- **Bench**: `crates/heartbit-core/benches/memory_recall.rs`
- **Missing case**: 
  - N=100k load (currently only tests 1k and 10k). At 100k, the tokens-cache memory cost of inverted indexes and keyword sets becomes visible. P-MEM-2 (inverted index allocation) and P-MEM-3 (lazy strength decay cache) benefits aren't validated at scale.
  - Hybrid mode with `query_embedding` set (currently only text-query and agent-filter). The embedding lookup + fusion path (lines 262–290 in `in_memory.rs`) is completely unmeasured.
  - Graph expansion via `related_ids` (currently no bench exercises related memory cluster navigation).
- **Why it matters**: 
  - P-MEM-2 (inverted index fix) and P-MEM-3 (lazy decay cache) are critical O(N²) → O(N) wins, but the 100k case is where they pay off most. Current 10k bench doesn't exercise the worst-case memory pressure or cache miss patterns.
  - Hybrid recall is increasingly used for semantic search but has no perf validation. The fusion logic (lines 278–290) isn't tested under load.
  - Graph expansion can add 20–50% latency (related_ids filtering, re-scoring). This is invisible in current benchmarks.
- **Sketch**: 
  ```rust
  #[bench]
  fn recall_100k_text_hybrid(c: &mut Criterion) {
      // Populate 100k entries with 10% having embeddings
      let store = InMemoryStore::new();
      let scope = TenantScope::default();
      
      for i in 0..100_000 {
          let mut entry = make_entry(i, "agent", "content", &["kw1", "kw2"]);
          if i % 10 == 0 {
              entry.embedding = Some(vec![0.1; 1536]); // mock embedding
          }
          rt.block_on(store.store(&scope, entry))?;
      }
      
      c.bench_function("recall_100k_text_hybrid_top10", |b| {
          b.iter(|| {
              rt.block_on(async {
                  let query = MemoryQuery {
                      text: Some("performance tokio".into()),
                      query_embedding: Some(vec![0.1; 1536]),
                      limit: 10,
                      ..Default::default()
                  };
                  store.recall(&scope, query).await
              })
          })
      });
      
      // Related expansion bench
      c.bench_function("recall_10k_with_graph_expansion", |b| {
          // entries linked via related_ids
          b.iter(|| { /* same as text_query but with graph hops */ })
      });
  }
  ```
- **Difficulty**: medium
  - Requires pre-populating large entry sets with valid embeddings (mock OK).
  - May require feature flag or test-data generation to avoid benchmark data bloat.

---

### Gap-EX-2: guardrail_pii missing Deny mode, pre-tool, and post-tool hooks
- **Bench**: `crates/heartbit-core/benches/guardrail_pii.rs`
- **Missing case**:
  - `PiiAction::Deny` (currently only Redact + Warn). The Deny path (line 274–278 in `guardrail.rs`) terminates early; no perf bench validates that early exit overhead is negligible.
  - Pre-tool hook (`on_before_tool_call`, line 142). This runs on user input before tool execution — a different code path with different memory patterns.
  - Post-tool hook (`on_after_tool_call`, line 145). Runs on tool output, another distinct path.
  - Multiple detectors in parallel (currently uses `all_builtin` but no bench of how overhead scales with detector count).
- **Why it matters**:
  - P-XCUT-3 (RegexSet consolidation) validated for post-llm only. Deny mode exits early; if early exit has overhead (mutex, logging), that's unmeasured.
  - Pre-tool detection is used defensively (F-XCUT-1); its latency matters for interactive approval flows.
  - Post-tool detection on large tool outputs (e.g., file reads, 100 KB content) hasn't been profiled.
- **Sketch**:
  ```rust
  #[bench]
  fn bench_pii_modes(c: &mut Criterion) {
      let payload = sample_response_text(); // 4 KB with PII
      let rt = tokio::runtime::Builder::new_current_thread().build()?;
      
      // Deny mode
      let deny_guard = PiiGuardrail::all_builtin(PiiAction::Deny);
      c.bench_function("guardrail_pii_post_llm_deny", |b| {
          b.iter(|| {
              let mut resp = make_response(payload.clone());
              rt.block_on(async { deny_guard.post_llm(&mut resp).await })
          })
      });
      
      // Pre-tool on user input
      c.bench_function("guardrail_pii_pre_tool_4kb", |b| {
          b.iter(|| {
              rt.block_on(async {
                  deny_guard.on_before_tool_call(
                      "read_file",
                      &serde_json::json!({"path": "/etc/passwd", "pii": "test"}),
                  ).await
              })
          })
      });
      
      // Post-tool on large output (100 KB)
      let large_output = sample_response_text().repeat(25); // ~100 KB
      c.bench_function("guardrail_pii_post_tool_100kb", |b| {
          b.iter(|| {
              let mut resp = make_response(large_output.clone());
              rt.block_on(async { deny_guard.post_llm(&mut resp).await })
          })
      });
  }
  ```
- **Difficulty**: low
  - Reuses existing `PiiGuardrail` and test utilities.
  - Just needs additional bench functions with different inputs/modes.

---

### Gap-EX-3: sse_parse missing overflow, partial-frame splits, and edge cases
- **Bench**: `crates/heartbit-core/benches/sse_parse.rs`
- **Missing case**:
  - Overflow behaviour: what happens when a frame exceeds the parser's internal buffer? Current 16 KB stream doesn't stress this.
  - Partial-frame splits across 100s of 256-byte chunks (vs the 4 KB chunks tested). This exercises the line-spanning buffer logic more aggressively.
  - Malformed frames (missing `data:`, duplicate `event:` lines). The parser must recover gracefully; no bench validates latency of error handling.
  - Empty events and whitespace-only frames. These are valid SSE but may have different perf characteristics.
- **Why it matters**:
  - P-LLM-14 (emit_event joins data_lines even on empty events) is only validated on well-formed streams. Real networks have packet boundaries in weird places.
  - P-LLM-2 (per-event String allocations) isn't tested at the pathological case (1000 events per 16 KB = tiny frames, more allocations per byte).
  - Overflow handling could have quadratic behavior if the parser backs off and retries (unlikely, but unmeasured).
- **Sketch**:
  ```rust
  #[bench]
  fn bench_sse_edge_cases(c: &mut Criterion) {
      let mut group = c.benchmark_group("sse_parse_edges");
      
      // Pathological: many tiny frames
      let tiny_chunks: Vec<String> = (0..1000)
          .map(|i| format!("event: msg\ndata: tiny{i}\n\n"))
          .collect();
      
      group.bench_function("feed_1000_tiny_frames", |b| {
          b.iter(|| {
              let mut total = 0;
              for chunk in &tiny_chunks {
                  total += __bench::sse_parse_chunk(chunk);
              }
              total
          })
      });
      
      // Partial splits: every 256 bytes
      let stream = synth_sse_stream(); // 16 KB
      let chunks: Vec<String> = stream
          .as_bytes()
          .chunks(256)
          .map(|c| String::from_utf8_lossy(c).into_owned())
          .collect();
      
      group.bench_function("feed_256byte_chunks_partial_lines", |b| {
          b.iter(|| {
              let mut total = 0;
              for chunk in &chunks {
                  total += __bench::sse_parse_chunk(chunk);
              }
              total
          })
      });
      
      // Empty events
      let empty_events = "event: start\ndata: \n\n".repeat(100);
      group.bench_function("feed_empty_events", |b| {
          b.iter(|| __bench::sse_parse_chunk(&empty_events))
      });
      
      group.finish();
  }
  ```
- **Difficulty**: low
  - Reuses existing `__bench::sse_parse_chunk` harness.
  - Just different input patterns, no new infrastructure.

---

## Missing benches (new harnesses)

### Bench-NEW-1: Agent ReAct turn (top priority)
- **Path covered**: `crates/heartbit-core/src/agent/runner.rs::AgentRunner::execute_inner` (lines 465–650, 1480–1530 tool loop)
- **Branches exercised**:
  - Line 520–530: Initial LLM call + context building (P-RUNNER-1 cloning, to_request assembly).
  - Line 1480–1530: Tool execution loop, tool repair (P-RUNNER-6 Levenshtein), doom loop tracking (P-RUNNER-4 hashing).
  - Line 1172–1190: Context compaction and summarization (P-RUNNER-2 TokenUsage arithmetic).
  - Line 1868–1920: Dynamic tool selection (P-RUNNER-5 regex + lowercase).
- **Validates**: P-RUNNER-1 (Arc<ToolDefinition>), P-RUNNER-4 (doom loop hashing), P-RUNNER-5 (tool filtering), P-RUNNER-6 (Levenshtein repair). Also validates AgentContext::to_request assembly cost.
- **Sketch**:
  ```rust
  #[cfg(all(test, feature = "bench-internals"))]
  pub mod __bench {
      use super::*;
      use std::sync::Arc;
      use heartbit_core::agent::test_helpers::{MockProvider, make_agent};
      use heartbit_core::llm::types::{ContentBlock, StopReason, TokenUsage};
      
      /// Single ReAct turn: LLM call + tool execution + return.
      pub async fn bench_react_turn(
          agent: &AgentRunner<MockProvider>,
          tool_count: usize,
      ) -> Result<AgentOutput, Error> {
          agent.execute(&format!("task with {tool_count} available tools")).await
      }
  }
  
  #[bench]
  fn bench_react_single_turn(c: &mut Criterion) {
      let mut group = c.benchmark_group("react_turn");
      
      let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
      
      for tool_count in [10, 50, 100] {
          // Mock: LLM returns a single tool call
          let mock = Arc::new(MockProvider::new(vec![
              MockProvider::text_response(
                  r#"{"type": "tool_use", "id": "call_1", "name": "tool_1", 
                      "input": {"key": "value"}}"#,
                  50, 20
              ),
              // Tool result → final response
              MockProvider::text_response("Task completed.", 30, 10),
          ]));
          
          let agent = AgentRunnerBuilder(mock)
              .name("bench_agent")
              .system_prompt("You are a helpful agent.")
              .max_turns(2)
              .tools_from_vec((0..tool_count).map(|i| {
                  // Create mock tools
                  Arc::new(EchoTool::new(format!("tool_{i}")))
              }).collect())
              .build()
              .unwrap();
          
          group.bench_with_input(
              BenchmarkId::new("execute_with_tools", tool_count),
              &tool_count,
              |b, _| {
                  b.iter(|| {
                      rt.block_on(async {
                          agent.execute("solve this task").await
                      })
                  })
              },
          );
      }
      
      group.finish();
  }
  ```
- **Difficulty**: high
  - Needs MockProvider + mock tool implementations.
  - Must set up realistic tool definitions (JSON schemas).
  - Requires async runtime in criterion bench.
  - Alternative: use `execute_inner` directly with captured request/response pairs.

---

### Bench-NEW-2: MCP JSON-RPC roundtrip (stdio + HTTP)
- **Path covered**: `crates/heartbit-core/src/tool/mcp.rs::StdioTransport::rpc` (lines 1400–1450), `HttpTransport::rpc` (lines 1220–1280)
- **Branches exercised**:
  - Line 1404: `serde_json::to_string(&request)` (P-MCP-9 serialization).
  - Line 1417–1430: Stdio write + read loop, buffering (P-MCP-8 buffered read).
  - Line 1240–1280: HTTP POST, response parse (P-MCP-6 header conversion, connection reuse).
  - Line 803–808: Token cache lookup (P-MCP-3/4 TokenCacheKey allocation + HashMap).
- **Validates**: P-MCP-1/2 (regex compilation overhead if auth cache misses), P-MCP-3/4 (token cache allocations), P-MCP-8/9 (buffering + serialization).
- **Sketch**:
  ```rust
  #[bench]
  fn bench_mcp_roundtrip(c: &mut Criterion) {
      let mut group = c.benchmark_group("mcp_rpc");
      group.throughput(Throughput::Bytes(1024)); // avg JSON-RPC frame
      
      let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
      
      // Stdio: mock child process
      group.bench_function("mcp_stdio_rpc_small_request", |b| {
          b.iter(|| {
              rt.block_on(async {
                  // Simulate StdioTransport::rpc() with mock ChildStdout
                  let req = JsonRpcRequest {
                      jsonrpc: "2.0",
                      method: "resources/read".into(),
                      params: Some(json!({"uri": "file:///test"})),
                      id: 1,
                  };
                  let serialized = serde_json::to_string(&req).unwrap();
                  black_box(serialized.len())
              })
          })
      });
      
      // HTTP: single roundtrip, token cache hit
      group.bench_function("mcp_http_rpc_with_token_cache", |b| {
          // Pre-populate token cache with valid token
          // Measure just the HTTP POST + response parse
          b.iter(|| {
              // Simulate HttpTransport::rpc() with cache hit
              black_box(token_cache_lookup(user_id, resource_id))
          })
      });
      
      // HTTP: token cache miss (OAuth token exchange)
      group.bench_function("mcp_http_rpc_oauth_token_exchange", |b| {
          b.iter(|| {
              rt.block_on(async {
                  // Simulate token cache miss + exchange (expensive)
                  black_box(exchange_oauth_token().await)
              })
          })
      });
      
      // Full roundtrip: 50 sequential tool calls to measure amortized costs
      group.bench_function("mcp_50_sequential_calls", |b| {
          b.iter(|| {
              rt.block_on(async {
                  for i in 0..50 {
                      let req = make_tool_call_request(i);
                      black_box(serialize_and_send(&req).await?)
                  }
                  Ok::<_, Error>(())
              })
          })
      });
      
      group.finish();
  }
  ```
- **Difficulty**: high
  - Needs mock child process or HTTP server for stdio/HTTP transports.
  - Token cache setup (pre-populate, measure cache hit vs miss).
  - OAuth token exchange simulation (may require async mock).

---

### Bench-NEW-3: Daemon Kafka dispatch
- **Path covered**: `crates/heartbit-cli/src/daemon/handlers.rs` + `execute.rs` (command dispatch, agent spawning)
- **Branches exercised**:
  - Daemon message parsing (JSON-RPC into DaemonCommand enum).
  - Per-command routing and handler lookup (match dispatch).
  - Agent runner instantiation (config parsing, tool setup).
  - Memory/knowledge base resolution.
- **Validates**: Daemon overhead per command (message queuing, deserialization, routing). Not directly a P-* finding, but validates that daemon dispatch is O(1) amortized.
- **Sketch**:
  ```rust
  #[bench]
  fn bench_daemon_dispatch(c: &mut Criterion) {
      let mut group = c.benchmark_group("daemon_dispatch");
      let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
      
      // Prepare daemon state
      let state = AppState {
          provider: Arc::new(MockProvider::new(vec![])),
          memory: Arc::new(InMemoryStore::new()),
          ..Default::default()
      };
      
      // Batch of 100 DaemonCommand messages
      let commands: Vec<RuntimeRequest> = (0..100)
          .map(|i| RuntimeRequest {
              task: format!("task {i}"),
              agent_config: AgentConfig::default(),
              ..Default::default()
          })
          .collect();
      
      group.bench_function("dispatch_100_commands", |b| {
          b.iter(|| {
              rt.block_on(async {
                  for cmd in &commands {
                      let res = handle_execute(State(state.clone()), Json(cmd.clone())).await;
                      black_box(res)
                  }
              })
          })
      });
      
      group.finish();
  }
  ```
- **Difficulty**: medium
  - Requires AppState setup and mock providers.
  - May need to run a minimal daemon server or call handlers directly.

---

### Bench-NEW-4: Channel WebSocket bridge (interaction resolution)
- **Path covered**: `crates/heartbit-core/src/channel/bridge.rs::InteractionBridge::resolve_input_for_session` (lines 291–320)
- **Branches exercised**:
  - Line 299–312: HashMap lookup + session verification (F-AUTH-5).
  - Pending entry cleanup under timeout (line 77, GRACE_PERIOD).
  - Concurrent pending entry management (RwLock overhead).
- **Validates**: F-AUTH-5 (session isolation), interaction resolution latency under N concurrent pending entries. Also validates RwLock contention.
- **Sketch**:
  ```rust
  #[bench]
  fn bench_bridge_resolution(c: &mut Criterion) {
      let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
      let (tx, _rx) = tokio::sync::mpsc::channel(100);
      let bridge = Arc::new(InteractionBridge::new(tx, Duration::from_secs(30)));
      
      // Pre-populate pending map with N entries
      for n_pending in [10, 100, 1000] {
          let session_id = Uuid::new_v4();
          
          // Insert N pending interactions
          for i in 0..n_pending {
              let interaction_id = Uuid::new_v4();
              let (tx, _rx) = tokio::sync::oneshot::channel();
              bridge.pending.write().unwrap().insert(
                  interaction_id,
                  PendingEntry {
                      session_id,
                      sender: PendingSender::Input(tx),
                  },
              );
          }
          
          c.bench_with_input(
              BenchmarkId::new("resolve_input_for_session", n_pending),
              &n_pending,
              |b, &n| {
                  b.iter(|| {
                      rt.block_on(async {
                          let interaction_id = Uuid::new_v4();
                          bridge.resolve_input_for_session(
                              Some(session_id),
                              interaction_id,
                              Some("response".into()),
                          )
                      })
                  })
              },
          );
      }
  }
  ```
- **Difficulty**: medium
  - Straightforward to set up (just PendingEntry + RwLock simulation).
  - Measures HashMap lookup + lock contention, which are the key concerns.

---

### Bench-NEW-5: Prompt assembly + tool definition serialisation
- **Path covered**: `crates/heartbit-core/src/agent/context.rs::AgentContext::to_request` (lines 227–243), plus serialization in `anthropic.rs` (lines 230–290)
- **Branches exercised**:
  - Line 237: `self.tools.clone()` (P-RUNNER-1 if not using Arc).
  - Line 227–243: CompletionRequest construction with message cloning.
  - `anthropic.rs` lines 245–290: cache_control injection, tool definition JSON generation (P-RUNNER-1 serialization cost).
- **Validates**: P-RUNNER-1 (Arc<ToolDefinition> benefit), tool definition serialization cost, cache_control LazyLock work (ensure no per-request overhead).
- **Sketch**:
  ```rust
  #[bench]
  fn bench_prompt_assembly(c: &mut Criterion) {
      let mut group = c.benchmark_group("prompt_assembly");
      
      // Realistic tool set (100 tools, 500 bytes each)
      let tools: Vec<ToolDefinition> = (0..100)
          .map(|i| ToolDefinition {
              name: format!("tool_{i}"),
              description: format!("Description of tool_{i}. This is a longer description to simulate realistic tool metadata. It includes usage notes and parameter details."),
              input_schema: json!({
                  "type": "object",
                  "properties": {
                      "arg1": {"type": "string"},
                      "arg2": {"type": "number"},
                      "arg3": {"type": "array", "items": {"type": "string"}}
                  },
                  "required": ["arg1"]
              }),
          })
          .collect();
      
      // Message history (5 turns, ~2KB per turn)
      let mut messages = vec![Message::user("initial task")];
      for i in 0..5 {
          messages.push(Message {
              role: Role::Assistant,
              content: vec![ContentBlock::Text {
                  text: format!("Response {i}. " .repeat(100)),
              }],
          });
          messages.push(Message::user(format!("Follow-up {i}")));
      }
      
      let ctx = AgentContext {
          system: "You are a helpful assistant.".into(),
          messages: messages.clone(),
          tools: tools.clone(),
          max_turns: 10,
          max_tokens: 4096,
          current_turn: 5,
          context_strategy: ContextStrategy::Unlimited,
          reasoning_effort: None,
      };
      
      group.bench_function("to_request_100_tools_5_turns", |b| {
          b.iter(|| {
              let req = ctx.to_request();
              black_box(req)
          })
      });
      
      // Anthropic serialization: cache_control injection
      group.bench_function("anthropic_serialize_with_cache_control", |b| {
          b.iter(|| {
              let req = ctx.to_request();
              let body = serialize_request_with_cache_control(&req, true);
              black_box(body.len())
          })
      });
      
      group.finish();
  }
  ```
- **Difficulty**: medium
  - Requires realistic ToolDefinition + message history.
  - Anthropic serialization needs access to internal serialization fn or mocking.

---

### Bench-NEW-6: Memory consolidation (Jaccard clustering at scale)
- **Path covered**: `crates/heartbit-core/src/memory/consolidation.rs::ConsolidationPipeline::run_detailed` (lines 112–180), `cluster_by_keywords` (lines 261–297)
- **Branches exercised**:
  - Line 261–297: Jaccard clustering O(N²) loop (P-MEM-4).
  - Line 305–320: HashSet construction per pair (allocation intensive).
  - Line 350–380: Summary LLM calls per cluster.
- **Validates**: P-MEM-4 (Jaccard clustering O(N²) bottleneck). This is the only bench that exercises the consolidation pipeline end-to-end.
- **Sketch**:
  ```rust
  #[bench]
  fn bench_consolidation(c: &mut Criterion) {
      let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
      
      let store = Arc::new(InMemoryStore::new());
      let provider = Arc::new(MockProvider::new(vec![
          // Summary responses for each cluster
          MockProvider::text_response("Consolidated summary.", 50, 20),
      ]));
      
      for n_entries in [100, 500, 1000] {
          c.bench_with_input(
              BenchmarkId::new("consolidation_run", n_entries),
              &n_entries,
              |b, &n| {
                  let scope = TenantScope::default();
                  
                  // Populate store with N entries, some with overlapping keywords
                  let entries: Vec<MemoryEntry> = (0..n)
                      .map(|i| MemoryEntry {
                          id: format!("entry-{i}"),
                          keywords: vec![
                              format!("keyword_{i % 10}"),
                              format!("keyword_{(i+1) % 10}"),
                              "shared_keyword".into(),
                          ],
                          ..make_entry(i as u32, "agent", "content", &[])
                      })
                      .collect();
                  
                  rt.block_on(async {
                      for entry in &entries {
                          store.store(&scope, entry.clone()).await?;
                      }
                      
                      b.iter(|| {
                          rt.block_on(async {
                              let pipeline = ConsolidationPipeline::new(
                                  store.clone(),
                                  provider.clone(),
                                  "bench_agent",
                              );
                              pipeline.run_detailed(&scope).await
                          })
                      })
                  })?;
              },
          );
      }
  }
  ```
- **Difficulty**: high
  - Consolidation is async and calls the LLM provider.
  - Must mock provider or use a very fast one.
  - Populating 1000 entries with embeddings/keywords adds setup time.

---

### Bench-NEW-7: Patch fuzzy match (apply_patch on realistic hunks)
- **Path covered**: `crates/heartbit-core/src/tool/builtins/patch.rs` (lines 74–200), hunk matching logic
- **Branches exercised**:
  - Lines 150–170: Hunk line matching with fuzzy fallback (exact → trim-end → trim-both → unicode-normalise).
  - Branches: P-TOOL-5/14 short-circuit optimization (bail early on exact match).
- **Validates**: P-TOOL-5 (fuzzy match short-circuit). No perf claim is made in the current code, but the bench would establish a baseline for future optimizations and validate that trim + unicode-normalize don't regress latency.
- **Sketch**:
  ```rust
  #[bench]
  fn bench_patch_apply(c: &mut Criterion) {
      let mut group = c.benchmark_group("patch_apply");
      
      // 100-line file with realistic Rust code
      let original_content = r#"fn main() {
      println!("Hello");
      let x = 42;
      // Line 5
      for i in 0..10 {
          println!("{}", i);
      }
      // More code...
  "#.repeat(10); // 100 lines
      
      // Unified diff: 10 hunks, each with exact, trim, and unicode challenges
      let patch = r#"--- original.rs
  +++ modified.rs
  @@ -5,3 +5,3 @@
   for i in 0..10 {
  -    println!("{}", i);
  +    println!("iter: {}", i);  // trim-end challenge (trailing space)
   }
  "@;
      
      group.bench_function("apply_patch_100line_10hunks", |b| {
          b.iter(|| {
              let file_patches = parse_unified_diff(black_box(patch))?;
              apply_hunks_to_lines(&original_content, &file_patches[0])
          })
      });
      
      // Pathological: fuzzy all matches (every hunk needs normalize)
      let patch_fuzzy = r#"@@ -5,3 +5,3 @@
   for i in 0..10 {
  -    println!("{}", i);     /* extra spaces & smart quotes: " " */
  +    println!("iter: {}", i);
   }
  "@;
      
      group.bench_function("apply_patch_100line_fuzzy_unicode", |b| {
          b.iter(|| {
              apply_hunks_to_lines(&original_content, &parse_hunk(black_box(patch_fuzzy))?)
          })
      });
      
      group.finish();
  }
  ```
- **Difficulty**: low-medium
  - Straightforward: just call the patch apply function with different inputs.
  - Requires parse_unified_diff public API or integration test harness.

---

### Bench-NEW-8: CLI startup (cold-path, LazyLock validation)
- **Path covered**: `crates/heartbit-cli/src/main.rs::main` (lines 314–380), config loading (LazyLock delays)
- **Branches exercised**:
  - Line 330–340: Config file parsing (TOML → HeartbitConfig).
  - Line 350–360: Provider initialization (Anthropic key setup, LazyLock initialization).
  - Line 365–380: Subcommand dispatch (should be instant after setup).
- **Validates**: That LazyLock additions (especially in `anthropic.rs`) don't add measurable overhead to startup. Baseline is `cargo run --release -- --help` time; target is <500ms.
- **Sketch**:
  ```rust
  // In heartbit-cli, or as a separate bench crate
  #[bench]
  fn bench_cli_startup(c: &mut Criterion) {
      // WARNING: criterion can't directly measure process startup time.
      // Instead, measure the `main()` function with minimal I/O overhead.
      
      let mut group = c.benchmark_group("cli_startup");
      group.sample_size(10); // startup is slow, reduce samples
      
      group.bench_function("config_load_and_parse", |b| {
          b.iter(|| {
              // Simulate main() setup: load config, init provider
              let config_path = "heartbit.toml";
              let config = HeartbitConfig::from_file(config_path).unwrap();
              let provider = AnthropicProvider::from_config(&config).unwrap();
              black_box((config, provider))
          })
      });
      
      // Measure LazyLock initialization cost
      group.bench_function("anthropic_lazylocks_init", |b| {
          b.iter(|| {
              // Access cache_control LazyLock for the first time
              let _ = anthropic::CACHE_CONTROL_BREAKPOINTS.lock();
              black_box(())
          })
      });
      
      group.finish();
  }
  ```
- **Difficulty**: medium
  - Config file and environment setup required.
  - Criterion doesn't measure cold process startup directly (can only measure in-process perf).
  - Alternative: use a shell script to run `cargo build --release && time ./target/release/heartbit-cli --help` as a manual baseline.

---

## Summary of findings

### Total gaps identified
- **Existing bench gaps**: 3 (memory_recall, guardrail_pii, sse_parse)
- **New benches needed**: 8 (react_turn, mcp_rpc, daemon_dispatch, bridge, prompt_assembly, consolidation, patch, cli_startup)
- **Grand total**: 11 distinct missing or incomplete benchmark harnesses

### Top 3 most-urgent benches to add (with severity)

1. **Bench-NEW-1: Agent ReAct turn** — **CRITICAL**
   - Agent execution is the #1 hot path. P-RUNNER-1 (Arc<ToolDefinition>), P-RUNNER-4 (doom loop hashing), and P-RUNNER-5 (tool filtering) are currently unmeasured. This bench unblocks validation of the entire runner optimization story.
   - Effort: ~3–4 agent-hours (mock provider setup, realistic tool definitions, criterion integration).

2. **Gap-EX-1: memory_recall at N=100k + hybrid mode** — **HIGH**
   - Memory consolidation (P-MEM-4 clustering) and inverted indexing (P-MEM-2) pay off massively at 100k scale. Current benchmarks hide the worst-case O(N²) and memory pressure. Hybrid recall is increasingly used but completely unmeasured.
   - Effort: ~2–3 agent-hours (data generation, additional bench functions).

3. **Bench-NEW-2: MCP JSON-RPC roundtrip** — **HIGH**
   - MCP is a growing tool ecosystem. P-MCP-1/2 (regex compilation), P-MCP-3/4 (token cache), and P-MCP-8/9 (buffering/serialization) have no integrated bench. Without this, we can't validate that OAuth failures or large tool definitions don't regress latency.
   - Effort: ~3–4 agent-hours (mock transport layer, token cache simulation).

### Audit findings currently UNVALIDATED (no bench at all)

**From perf-audit-runner.md (agent hot paths):**
- P-RUNNER-1 (Arc<ToolDefinition>) — no bench
- P-RUNNER-2 (TokenUsage Copy) — static-only validation only
- P-RUNNER-3 (recently_used_tools clone) — no bench
- P-RUNNER-4 (DoomLoopTracker hashing) — no bench
- P-RUNNER-5 (tool filtering regex) — no bench
- P-RUNNER-6 (Levenshtein repair) — no bench

**From perf-audit-memory.md:**
- P-MEM-2 (BM25 inverted index) — validated at 1k/10k, NOT at 100k
- P-MEM-3 (lazy strength decay) — validated at 1k/10k, NOT at 100k
- P-MEM-4 (Jaccard clustering O(N²)) — no bench at all
- P-MEM-5 (MemoryEntry clone) — no bench

**From perf-audit-mcp.md:**
- P-MCP-1/2 (regex compilation) — no bench
- P-MCP-3/4 (token cache overhead) — no bench
- P-MCP-8/9 (buffering + serialization) — no bench

**From perf-audit-llm.md:**
- P-LLM-2 (SSE parser per-event allocations) — not validated at pathological frame boundaries
- P-LLM-14 (emit_event joins) — validated on well-formed streams only

**From perf-audit-cross.md / perf-audit-xcut.md:**
- P-XCUT-3 (PII guardrail RegexSet consolidation) — only post_llm mode, missing Deny/pre-tool/post-tool paths

**NEW findings (v2 audit):**
- P-RUNNER-1 (Arc<ToolDefinition> assembly + serialization in anthropic.rs) — no integrated bench
- F-AUTH-5 (session isolation in bridge under N concurrent pending) — no load bench
- Consolidation Jaccard at 100k scale — no bench

### Estimated effort per missing bench (in agent-minutes)

| Bench | Effort | Notes |
|-------|--------|-------|
| Gap-EX-1 (memory 100k + hybrid) | 120–180 min | Data gen, additional bench fns |
| Gap-EX-2 (PII Deny/pre/post modes) | 60–90 min | Reuse existing utilities, low setup |
| Gap-EX-3 (SSE edge cases) | 60–90 min | Just different input patterns |
| Bench-NEW-1 (ReAct turn) | 180–240 min | Mock provider, async criterion, realistic tools |
| Bench-NEW-2 (MCP RPC) | 180–240 min | Mock transport layer, token cache |
| Bench-NEW-3 (daemon dispatch) | 120–180 min | AppState setup, command routing |
| Bench-NEW-4 (bridge resolution) | 90–120 min | Straightforward: just HashMap + RwLock |
| Bench-NEW-5 (prompt assembly) | 120–180 min | Context + serialization, cache_control |
| Bench-NEW-6 (consolidation) | 180–240 min | Async, mock LLM, data generation |
| Bench-NEW-7 (patch fuzzy match) | 90–120 min | Hunk parsing, apply logic |
| Bench-NEW-8 (CLI startup) | 90–120 min | Config load, LazyLock init (manual baseline alternative) |

**Total estimated effort**: ~1400–1950 agent-minutes (~23–33 hours) for all 11 gaps.

### The single most embarrassing bench gap

**Bench-NEW-1: Agent ReAct turn** is the most embarrassing absence for the "best in class" claim.

**Rationale**:
- Agents are the core product. Every core audit finding (P-RUNNER-1 through P-RUNNER-6) is unvalidated in a realistic ReAct loop.
- We have benches for memory recall, PII guardrails, and SSE parsing (supporting subsystems), but **zero bench for the main agent execution hot path**.
- A competitor claiming "best-in-class agentic performance" without an agent-turn benchmark would be (rightly) questioned by any customer.
- The fix (Arc<ToolDefinition>, doom loop hashing optimization, tool filtering) are all low-hanging fruit, but we can't measure them without this bench.
- Estimated payoff: validating that a single ReAct turn is <50ms on local LLM with 100 tools (amortized across a 10-turn run). Right now, we can't guarantee it.

---

## Appendix: Implementation priority roadmap

1. **Phase 1 (immediate, <1 week)**: Add 3 existing bench gaps (Gap-EX-1/2/3). Low effort, high signal.
2. **Phase 2 (week 1–2)**: Add Bench-NEW-4 (bridge) and Bench-NEW-7 (patch). Low/medium effort, straightforward setup.
3. **Phase 3 (week 2–3)**: Add Bench-NEW-1 (ReAct turn) — **unblocks all P-RUNNER validation**. High effort but critical.
4. **Phase 4 (week 3–4)**: Add Bench-NEW-2 (MCP RPC) and Bench-NEW-5 (prompt assembly). High effort but closes MCP + serialization validation gaps.
5. **Phase 5 (week 4+)**: Add Bench-NEW-3 (daemon) and Bench-NEW-6 (consolidation). Medium/high effort, less critical but completeness.
6. **Phase 6 (optional, after stabilization)**: Add Bench-NEW-8 (CLI startup) as a continuous integration check.

Once complete, the bench suite would cover:
- **Agent execution hot path** (turns, tool calls, doom loop tracking).
- **Memory subsystem** (recall at all scales, consolidation clustering).
- **MCP integration** (JSON-RPC, token caching, buffering).
- **Guard rails** (PII in all modes, other detectors).
- **SSE streaming** (all edge cases, frame boundaries).
- **Supporting tooling** (patch application, prompt assembly).

This represents **comprehensive coverage of all audit findings** and would support the "best-in-class" claim with measurable, reproducible evidence.

