# Perf audit: MCP / A2A

## Executive Summary
Conducted comprehensive performance analysis of the MCP and A2A subsystems. Identified 12 findings across hot paths, token cache efficiency, regex compilation overhead, JSON-RPC serialization patterns, and connection pooling. No critical DoS vectors or throughput-breaking issues detected, but several warm-path optimizations available with moderate cost/benefit tradeoff.

## P-MCP-1 [Medium]: Regex compilation per `redact_idp_body` call
- **File**: crates/heartbit-core/src/tool/mcp.rs:421-445
- **Observation**: `redact_idp_body` compiles 3 regex patterns (JWT, Bearer, JSON token fields) on every IdP error response—no memoization via `LazyLock`/`OnceLock`. Each failed token exchange triggers 3 `Regex::new()` calls + NFA compilation.
- **Hypothesized cost**: ~500–800 µs per failed token exchange (3 regex compiles), worst-case. Compilation dominates if body is large (pattern matching cost is ~10–50 µs). Per-call if OAuth fails frequently (transient IdP issues, invalid tokens).
- **Frequency**: warm-path (on OAuth errors only, rare in steady state but critical in auth recovery loops)
- **Fix sketch**: Wrap the 3 patterns in a `lazy_static::Lazy<Vec<Regex>>` or `std::sync::OnceLock<Vec<Regex>>` at module level. Compile once at first call, reuse forever.
- **Security delta**: N/A (redaction logic unchanged)
- **Validation**: needs-bench (profile token-exchange failures to confirm 3× regex overhead)

## P-MCP-2 [High]: Regex compilation in `sanitize_log_field` at notification time
- **File**: crates/heartbit-core/src/tool/mcp.rs:235–250 + 421–446
- **Observation**: `handle_log_notification()` calls the nested `sanitize_log_field()` which manually loops chars. However, the sibling `redact_idp_body()` in the same call path compiles regexes. If logs with sensitive fields arrive frequently, the cumulative regex overhead becomes measurable.
- **Hypothesized cost**: ~500 µs per batch of logs (if 10+ log notifications with redaction per second). Char-loop cost ~1–5 µs (already good), but regex dominates.
- **Frequency**: hot-path (log notifications from MCP server may arrive during tool execution)
- **Fix sketch**: Pre-compile 3 regex patterns into a module-level `lazy_static::Lazy<Vec<Regex>>`. Keep char-loop for control-char sanitization (it's fine).
- **Security delta**: N/A (F-MCP-6 / F-MCP-16 unaffected)
- **Validation**: needs-bench (profile log notification handling under load)

## P-MCP-3 [Medium]: Per-call `TokenCacheKey` struct allocations
- **File**: crates/heartbit-core/src/tool/mcp.rs:803–808, 1005–1009, 1122–1125
- **Observation**: Every token lookup (both legacy and resource-scoped) constructs a new `TokenCacheKey { tenant_id: tenant_id.to_string(), user_id: user_id.to_string(), resource: String::new(), scopes: String::new() }`. Four `to_string()` calls per lookup, even when the cache read succeeds immediately. The 4-tuple prevents collisions (F-MCP-8), but construction cost is paid redundantly.
- **Hypothesized cost**: ~2–5 µs per token cache lookup (2× `String` allocations + 2× empty `String` allocs). Over 1000 tool calls/sec with per-tool auth, this is ~2–5 ms/sec aggregate.
- **Frequency**: hot-path-per-tool-call (every `TokenExchangeAuthProvider::auth_header_for_resource` call)
- **Fix sketch**: Change HashMap key type from `TokenCacheKey` to a borrowed `TokenCacheKeyRef<'a>` (using `Cow<str>` or lifetime refs), or switch to `FxHashMap` + custom hasher for faster hashing. Avoid `to_string()` in the lookup path by using `Cow::Borrowed` for the key. Trade-off: more complex key type but eliminates allocations.
- **Security delta**: N/A if key structure (4-tuple) is preserved. F-MCP-8 remains intact.
- **Validation**: static-only (Rust borrow checker; bench the HashMap lookups before/after)

## P-MCP-4 [High]: `HashMap<TokenCacheKey, _>` uses default hasher
- **File**: crates/heartbit-core/src/tool/mcp.rs:794
- **Observation**: Token cache uses `RwLock<HashMap<TokenCacheKey, (String, Instant)>>` with the default `DefaultHasher`. Hashing 4 strings per lookup. No explicit use of `FxHashMap` (faster for small integer/string keys) or custom `Hasher` (e.g., `AHashMap`).
- **Hypothesized cost**: ~50–100 ns per hash (4 string hashes), vs ~20–30 ns for `FxHashMap`. Over 10k tool calls/sec, this is ~0.3–0.7 ms/sec (measurable but not dominating).
- **Frequency**: hot-path-per-tool-call (on every token lookup, both cache hit and miss)
- **Fix sketch**: Replace `HashMap` with `FxHashMap` (from `rustc-hash` crate, commonly available). Drop-in replacement, 2–3× faster for string keys.
- **Security delta**: N/A (hash function is non-cryptographic by design; F-MCP-8 unaffected)
- **Validation**: static-only (drop-in; bench before/after)

## P-MCP-5 [Medium]: Per-call `String::clone()` in `mcp_tool_to_definition` and tool stamping
- **File**: crates/heartbit-core/src/tool/mcp.rs:398–412, 2425–2434
- **Observation**: `mcp_tool_to_definition()` clones `tool.name` and `tool.input_schema` on every call. When stamping tools from `McpTransportPool::tools_for_user()` (line 2431), this function is called once per tool per user—for 50 tools and 10 users, that's 500 clones of (name string + JSON schema). Each tool definition is then wrapped in a new `Arc<McpTool>` with the cloned definition stored.
- **Hypothesized cost**: ~10–20 µs per tool definition (string clone + JSON value clone). For 50 tools, ~500–1000 µs per `tools_for_user` call. If called once per user session (not per tool call), amortized low; if called per request, high.
- **Frequency**: warm-path-per-user-session (tools_for_user is called once during setup, not per tool call)
- **Fix sketch**: Store `ToolDefinition` by Arc in `PoolEntry.tools` instead of `McpToolDef`, or use `Arc<McpToolDef>` so cloning is cheap (Arc pointer copy). Alternatively, return a reference to the cached definition and only clone on demand.
- **Security delta**: N/A
- **Validation**: static-only (structure audit); needs-bench if called frequently

## P-MCP-6 [Low]: `content-type` header converted to String on every HTTP response
- **File**: crates/heartbit-core/src/tool/mcp.rs:1297–1302
- **Observation**: `response.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string()` allocates a new `String` for every HTTP response, even though only `.contains("text/event-stream")` is checked. The header is already a `&str`.
- **Hypothesized cost**: ~1–2 µs per HTTP request (String alloc + UTF-8 validation already done by `to_str()`).
- **Frequency**: hot-path-per-rpc (every MCP HTTP call)
- **Fix sketch**: Use `to_str()` directly without `.to_string()`, or use `.starts_with("text/event-stream")` on the `&str`. Eliminates allocation.
- **Security delta**: N/A
- **Validation**: static-only (trivial code change)

## P-MCP-7 [High]: No connection pooling / keep-alive for HTTP MCP clients
- **File**: crates/heartbit-core/src/tool/mcp.rs:884–889, 2004–2009
- **Observation**: Every `McpClient::connect_http()` constructs a new `reqwest::Client` with `reqwest::redirect::Policy::none()` (correct) but no explicit configuration of HTTP/2, keep-alive, or connection pool. The default reqwest client pools connections per task/thread, but the code doesn't leverage connection reuse—each `HttpTransport.rpc()` call may establish a new TCP connection to the same MCP server.
- **Hypothesized cost**: TCP handshake (SYN, SYN-ACK, ACK) ~50–100 ms on high-latency networks, TLS negotiation ~100–300 ms. If 100 calls/sec to the same MCP server over the Internet, this is 5–30 GB/sec of wasted handshakes. Even on localhost (~1 ms), 50–100 µs per call is measurable.
- **Frequency**: hot-path (every MCP HTTP RPC if connection reuse is broken)
- **Fix sketch**: Confirm that reqwest::Client pooling is working by checking if the same client instance is reused across calls. If not, ensure that `McpTransportPool.pool` stores the same `reqwest::Client` across all requests. Verify HTTP/2 is enabled (it should be default for HTTPS in reqwest 0.12). Consider explicit `.http2_prior_knowledge()` for HTTP endpoints.
- **Security delta**: N/A (no change to redirect policy or HTTPS enforcement)
- **Validation**: measured (enable reqwest debug logging or tcpdump to confirm TCP reuse)

## P-MCP-8 [Critical]: Manual buffered reading in `read_line_capped` may be inefficient
- **File**: crates/heartbit-core/src/tool/mcp.rs:515–560
- **Observation**: Implements manual buffered line reading with `fill_buf()` + `consume()` on `BufReader<ChildStdout>`. The loop accumulates bytes into a `String` buffer, clearing and re-appending per iteration. For large lines or slow readers, this incurs repeated `fill_buf()` syscalls and string reallocation. The check `if total.saturating_add(take) > max_bytes` (line 540) is correct for DoS prevention, but the implementation may read past line boundaries and then discard—wasteful.
- **Hypothesized cost**: For a 100 KB JSON-RPC frame, if `fill_buf()` returns 64 KB chunks, we loop 2–3 times, each with a `fill_buf()` syscall (~5–10 µs on Linux), plus string append (~2–5 µs per chunk). Total ~30–50 µs per large frame. Frames up to 512 KB: ~100–200 µs. This is measurable if MCP servers send large tool definitions or resource content.
- **Frequency**: hot-path (every stdio transport RPC response read)
- **Fix sketch**: Use `BufReader::read_line()` directly with the cap check after reading. Alternatively, use `tokio::io::AsyncReadExt::read_to_string()` with a pre-allocated buffer of `MCP_STDIO_LINE_MAX_BYTES` and a custom timeout. Measure the actual `fill_buf()` cost in your testbed.
- **Security delta**: F-MCP-4 (DoS cap at `MCP_STDIO_LINE_MAX_BYTES`) is preserved; the fix should not change the cap or introduce unbounded reads.
- **Validation**: measured (trace `fill_buf()` syscalls and string allocations during stdio transport RPC)

## P-MCP-9 [Medium]: `serde_json::to_string()` on every JSON-RPC request
- **File**: crates/heartbit-core/src/tool/mcp.rs:1404, 1436
- **Observation**: `StdioTransport::rpc()` and `notify()` call `serde_json::to_string(&request)` and `serde_json::to_string(&notification)` on every request, producing a heap-allocated `String`. For frequent small requests (e.g., ping, list), this is a few hundred bytes of allocation + heap free per call. No use of `bytes::Bytes` or stack-allocated buffers.
- **Hypothesized cost**: ~5–10 µs per JSON-RPC request (serialize + String alloc). For 1000 stdio calls/sec, ~5–10 ms/sec aggregate.
- **Frequency**: hot-path (every stdio RPC)
- **Fix sketch**: Use `serde_json::Serializer` to write directly to a reusable `Vec<u8>` buffer (or use `bytes::BytesMut`), avoiding the intermediate `String`. Or use a thread-local `String` buffer that's cleared and reused per request.
- **Security delta**: N/A
- **Validation**: static-only (refactor to use buffer; bench before/after)

## P-MCP-10 [Medium]: `tool.definition()` cloned on every `tools/list` response in MCP server
- **File**: crates/heartbit-core/src/tool/mcp_server.rs:332–338
- **Observation**: `handle_tools_list()` iterates tools and calls `.definition()` on each, which for `McpTool` returns `self.def.clone()` (cloning a `ToolDefinition` with name string + JSON schema). For 50 tools, this is 50 clones of potentially large JSON objects, even though they're cached in memory. The definitions are then serialized to JSON again in the response.
- **Hypothesized cost**: ~50–200 µs for 50 tools (50 clones of ToolDefinition + JSON serialization). Per `tools/list` call.
- **Frequency**: warm-path (tools/list may be called once per client session or per UI refresh, but not per tool call)
- **Fix sketch**: Cache the tool list response as a pre-serialized JSON string or `Value` at server initialization. Return a reference or re-use the cached JSON on every `tools/list` call. Only rebuild if tools are dynamically added/removed.
- **Security delta**: N/A
- **Validation**: static-only (cache response)

## P-MCP-11 [Medium]: Token cache lock held during async token exchange request
- **File**: crates/heartbit-core/src/tool/mcp.rs:1011–1016, 1082
- **Observation**: `TokenExchangeAuthProvider::auth_header_for_resource()` acquires a read lock on `token_cache` (line 1011–1016) to check for a cached token, which is correct. However, on a cache miss (lines 1031–1082), the write lock is acquired **after** the async `client.post().send().await` completes (line 1082). This is good. But if the read lock acquisition itself contends (many concurrent requests), the contention is serialized. The RwLock allows concurrent reads, but each read acquires the lock struct.
- **Hypothesized cost**: ~100–500 ns per lock acquisition on uncontended paths; ~10–50 µs if the lock is contended and multiple tasks are waiting. Contention risk is low if users are sparse, but high if 1000+ concurrent users calling tools per second.
- **Frequency**: hot-path-per-tool-call (every tool call with `TokenExchangeAuthProvider`)
- **Fix sketch**: Consider switching from `RwLock` to a specialized concurrent cache (e.g., `DashMap` or `parking_lot::RwLock` for slightly better performance on uncontended paths). Alternatively, use an atomic slot per tenant+user+resource and only lock on cache miss. For now, RwLock is acceptable unless lock contention is measured.
- **Security delta**: N/A
- **Validation**: measured (profile lock contention under 100+ concurrent user simulations)

## P-MCP-12 [Low]: `HandoffTool::definition()` clones cached definition on every call
- **File**: crates/heartbit-core/src/tool/handoff.rs:116–117
- **Observation**: `HandoffTool` caches the definition at construction (line 55–85) but `definition()` returns `self.cached_definition.clone()` (line 117), cloning a `ToolDefinition` with a description string that contains serialized JSON of all target agents.
- **Hypothesized cost**: ~5–20 µs per `definition()` call (String clone + JSON object clone). If `definition()` is called 10 times during setup, ~100–200 µs.
- **Frequency**: warm-path (definition() called during agent initialization, not per tool call)
- **Fix sketch**: Return a reference to the cached definition, or store the definition in an `Arc<ToolDefinition>` and clone the Arc (pointer copy) instead of the definition.
- **Security delta**: N/A
- **Validation**: static-only (refactor return type)

---

## Cross-cutting recommendations

### 1. Lazy-compile regex patterns
Add `regex::Regex` compilation to module-level `lazy_static::Lazy<Vec<Regex>>`:
- 3 patterns in `redact_idp_body()` (JWT, Bearer, JSON token fields)
- Estimated savings: **~500–800 µs per failed token exchange**

### 2. Use `FxHashMap` or `AHashMap` for token cache
Replace `HashMap<TokenCacheKey, _>` with `FxHashMap` from `rustc-hash`:
- Drop-in replacement, 2–3× faster for string-keyed lookups
- Estimated savings: **~20–50 ns per lookup, or ~0.3–0.7 ms/sec at 10k lookups/sec**

### 3. Buffer JSON-RPC serialization
Reuse a `Vec<u8>` or `String` buffer across stdio RPC calls instead of allocating per-call:
- Estimated savings: **~5–10 µs per stdio RPC**

### 4. Pre-serialize tool list response
Cache the MCP server's `tools/list` JSON response after initialization:
- Estimated savings: **~50–200 µs per tools/list call** (amortized over many clients)

### 5. Eliminate `to_string()` on header reads
Convert content-type header directly to `&str` without allocating:
- Trivial fix, estimated savings: **~1–2 µs per HTTP response**

---

## Findings summary
- **Total findings**: 12
- **Critical**: 1 (P-MCP-8: inefficient buffered line reading)
- **High**: 3 (P-MCP-2, P-MCP-4, P-MCP-7)
- **Medium**: 6 (P-MCP-1, P-MCP-3, P-MCP-5, P-MCP-6, P-MCP-10, P-MCP-11)
- **Low**: 2 (P-MCP-6, P-MCP-12)

### Top 3 quick wins
1. **Lazy-compile regex patterns** (P-MCP-1, P-MCP-2): ~500–800 µs per failed token exchange, simple `lazy_static` wrapper.
2. **Use `FxHashMap` for token cache** (P-MCP-4): 2–3× faster lookups, drop-in replacement.
3. **Buffer JSON-RPC serialization** (P-MCP-9): ~5–10 µs per stdio RPC, reuse buffer pool or thread-local.

### Rejected suggestions
- **Lifting `MCP_STDIO_LINE_MAX_BYTES` cap**: REJECTED. Cap is essential for F-MCP-4 (DoS prevention). P-MCP-8 suggests optimizing the implementation, not raising the limit.
- **Reverting `TokenCacheKey` 4-tuple to 2-tuple**: REJECTED. F-MCP-8 requires the 4-tuple to prevent collisions across resource/scope boundaries.
- **Skipping `sanitize_log_field` or `redact_idp_body`**: REJECTED. F-MCP-6 (log injection) and F-MCP-16 (token leak) require these defenses. P-MCP-1/P-MCP-2 suggest lazy-compiling the regex, not removing the sanitization.
- **Sharing `TokenCacheKey` across tenants**: REJECTED. F-MCP-8 / F-AUTH-5 requires tenant-aware key structure.

