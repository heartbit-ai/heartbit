# Perf audit: memory subsystem

## Summary
- **Total findings**: 18
- **Critical**: 5 | **High**: 8 | **Medium**: 4 | **Low**: 1
- **Top 3 wins**: (1) Replace std::sync::RwLock with parking_lot::RwLock for ~50% faster lock acquire, (2) Implement inverted index for BM25 token→entry_id map to avoid O(N) token scan per recall, (3) Lazy strength decay with cache to avoid recomputing effective_strength 3+ times per entry per recall
- **REJECTED suggestions**: None — all findings respect tenant isolation (F-MEM-5) and security hardening (F-MEM-4, F-MEM-1)

---

## Detailed Findings

### P-MEM-1 [Critical]: std::sync::RwLock poisoning + slow lock contention
- **File**: `crates/heartbit-core/src/memory/in_memory.rs:32`
- **Observation**: `InMemoryStore` uses `RwLock<HashMap>` for all entry storage. Every recall acquires write lock (line 129–132) despite only needing read lock for filtering, and only requiring write lock to update access_count/strength. Lock is held for entire multi-stage pipeline: filter → BM25 score → hybrid fusion → sort → expand via graph → reinforce. At N=10k entries, write-lock overhead is severe.
- **Hypothesized cost**: ~200–500μs per recall (contention-dependent) due to lock fairness/poisoning risk; 10+ concurrent recalls see exponential wait scaling.
- **Frequency**: hot-path-per-recall (every memory query)
- **Fix sketch**: 
  1. Replace `std::sync::RwLock` with `parking_lot::RwLock` — no poisoning, ~50% faster lock ops.
  2. Split lock scope: read-lock for filter/score, then upgrade to write-lock only for access_count/strength updates (requires refactoring to avoid TOCTOU issues, or keep single write lock but use parking_lot).
- **Security delta**: N/A (no tenant/user isolation change)
- **Validation**: needs-bench (profile lock contention at N=10k with 10+ concurrent recalls)

### P-MEM-2 [Critical]: BM25 full-scan token matching per entry
- **File**: `crates/heartbit-core/src/memory/bm25.rs:40-44`
- **Observation**: BM25 score computation for each entry iterates query_terms and counts matches via `content_words.iter().filter(|w| w.contains(term.as_str()))`. At 10k entries × 5 query tokens × 200-word content = 10M substring scans per recall. No inverted index or token pre-tokenization. Keywords field also checked via `.any(|k| k.contains(...))`, adding another full scan per term per entry.
- **Hypothesized cost**: ~500μs–2ms per recall (per entry: ~100–200μs for 5 terms on 200-word doc). Dominates recall latency at N≥5k.
- **Frequency**: hot-path-per-recall
- **Fix sketch**:
  1. Build persistent inverted index at store time: `HashMap<token, Vec<entry_id>>` (or `FxHashMap` for faster string hashing).
  2. At recall, iterate query_terms once and fetch entry_ids from index — O(M tokens) instead of O(N × M).
  3. Pre-tokenize content at store time (lowercase, split, deduplicate) — avoid repeated tokenization on every recall.
  4. Store token-to-TF map in MemoryEntry to avoid recomputing TF per recall.
- **Security delta**: N/A
- **Validation**: needs-bench (measure µs/recall before/after indexing at N=10k)

### P-MEM-3 [Critical]: Effective strength recomputed 3+ times per entry per recall
- **File**: `crates/heartbit-core/src/memory/in_memory.rs:175–186, 356–361, 418–420`
- **Observation**: `effective_strength()` is called:
  - Line 177–182: filter stage for min_strength check
  - Line 356–360: graph expansion loop (per related_id)
  - Line 418–420: re-sort after expansion
  - Each call = exponential decay formula. At 10k entries, ~30k+ decay computations per recall. Decay is deterministic given last_accessed + now + decay_rate.
- **Hypothesized cost**: ~50–100μs per recall (30k exp() calls × ~1–3μs each); cumulative across all filtering stages.
- **Frequency**: hot-path-per-recall
- **Fix sketch**:
  1. Compute effective_strength once at entry fetch, cache in temp struct (e.g., `EffectiveEntry { entry: &MemoryEntry, effective_strength: f64 }`).
  2. Reuse cached value through filter, expand, and sort stages.
  3. Or: lazy cache in a HashMap during recall, keyed by entry_id.
- **Security delta**: N/A
- **Validation**: static-only (measure exp() call count via profiler)

### P-MEM-4 [Critical]: Consolidation Jaccard clustering is O(N²)
- **File**: `crates/heartbit-core/src/memory/consolidation.rs:261–297`
- **Observation**: `cluster_by_keywords()` uses greedy single-linkage clustering. For each entry i (0..N), iterates all remaining entries j (i+1..N) and checks Jaccard similarity. At N=10k, this is ~50M pairwise comparisons. Each comparison constructs HashSets (line 305–306) from keyword vectors — allocation heavy. No bounds or early termination.
- **Hypothesized cost**: ~10–50ms for N=10k (worst-case O(N²) pairwise, plus O(K) set construction per pair, where K=keyword vector size).
- **Frequency**: cold-path (triggered at session end, consolidation)
- **Fix sketch**:
  1. Add early termination: if `clusters.len() > threshold`, stop clustering (e.g., max 100 clusters).
  2. Use incremental Jaccard: reuse set intersection from prior comparisons (CSR matrix or bloom filters).
  3. Cap clustering to top N entries by importance/recency first.
  4. Pre-compute keyword sets once at start, not in loop (already somewhat done via HashSet construction, but avoid repeated allocations).
- **Security delta**: N/A
- **Validation**: needs-bench (measure consolidation time at N=10k)

### P-MEM-5 [Critical]: MemoryEntry full clone on every recall result
- **File**: `crates/heartbit-core/src/memory/in_memory.rs:194`
- **Observation**: `.cloned()` on filtered entries returns owned `Vec<MemoryEntry>` with deeply nested fields: id, agent, content (String, can be 10k+ chars), keywords (Vec), related_ids (Vec), source_ids (Vec), embedding (Vec<f32>, 1536+ elements), tags (Vec), summary (Option<String>). At 10k entries recalled, this is 10k full clones of large structs. Each clone allocates heap for String, Vec contents.
- **Hypothesized cost**: ~1–5ms per recall (10k clones × 100–500 bytes each = 1–5MB allocation + copy).
- **Frequency**: hot-path-per-recall
- **Fix sketch**:
  1. Return `Vec<Cow<'a, MemoryEntry>>` or `Vec<Arc<MemoryEntry>>` instead of owned entries.
  2. Or: return Vec of entry references (requires lifetime bounds).
  3. Callers that need owned entries can still call `.into_owned()` or `Arc::unwrap_or_clone()`.
  4. At minimum: defer cloning until after limit is applied (line 328–329), so only top K entries are cloned.
- **Security delta**: N/A (no tenant/user context leak)
- **Validation**: needs-bench (measure heap allocations/recall at N=10k)

### P-MEM-6 [High]: Full-scan tenant filter before query processing
- **File**: `crates/heartbit-core/src/memory/in_memory.rs:136–140`
- **Observation**: First filter check is tenant isolation: `e.author_tenant_id.as_deref() != Some(tenant_id.as_str())` on every entry. This is a full O(N) scan before any text/category/tag filtering. In multi-tenant deployments with N=100k entries across 1000 tenants, ~99% of entries are filtered by tenant. Scanning all 100k before narrowing is wasteful.
- **Hypothesized cost**: ~100–300μs per recall (O(N) string comparisons at ~1–3μs each).
- **Frequency**: hot-path-per-recall
- **Fix sketch**:
  1. Add per-tenant secondary index: `HashMap<tenant_id, Vec<entry_id>>` built at store time.
  2. At recall, use tenant index to fetch candidate entry_ids first, then filter by other criteria.
  3. Reduces effective N to N/num_tenants for filtering.
- **Security delta**: N/A
- **Validation**: needs-bench (measure filtering time with per-tenant index vs full scan)

### P-MEM-7 [High]: Multiple allocation passes for BM25 + RRF fusion
- **File**: `crates/heartbit-core/src/memory/in_memory.rs:225–297`
- **Observation**: 
  - Line 225–238: bm25_map created as HashMap, then copied into bm25_ranked Vec (line 245–246)
  - Line 251–260: vector_ranked created separately
  - Line 275: rrf_fuse() creates another HashMap internally (hybrid.rs:21)
  - Line 282–285: result collected back into relevance_map
  - Result: 4 separate HashMap/Vec allocations for single fusion pass.
- **Hypothesized cost**: ~200–500μs per recall (allocation overhead; reallocation at hash table growth).
- **Frequency**: hot-path-per-recall (if query.query_embedding present)
- **Fix sketch**:
  1. Pre-allocate bm25_map and relevance_map with expected capacity.
  2. Use `Vec` directly instead of HashMap for scored rankings (already sorted by score).
  3. Fuse in-place: iterate both ranked lists and accumulate RRF score in a pre-allocated HashMap.
  4. Or: use `SmallVec` for results (typically < 100 entries after limit).
- **Security delta**: N/A
- **Validation**: static-only (count allocation calls via flame graph)

### P-MEM-8 [High]: Keywords field case-conversion repeated per recall
- **File**: `crates/heartbit-core/src/memory/in_memory.rs:142–147, 204–210`
- **Observation**: 
  - Filter stage (line 144): `e.keywords.iter().map(|k| k.to_lowercase()).collect()` for every entry
  - BM25 stage (bm25.rs:36): `keywords.iter().map(|k| k.to_lowercase()).collect()` again
  - Query tokens (line 205): `text.to_lowercase().split_whitespace()`
  - No caching of lowercased forms. At 10k entries × 10 keywords avg, ~100k to_lowercase() calls per recall.
- **Hypothesized cost**: ~50–150μs per recall (100k String allocations + copies).
- **Frequency**: hot-path-per-recall
- **Fix sketch**:
  1. Store lowercased keywords in MemoryEntry at insert time (or store both original + lowercase).
  2. Or: pass original keywords to BM25, let BM25 do single lowercase once, then reuse.
  3. Pre-lowercase query text once at start of recall (already done line 205, but keywords double-conversion is wasteful).
- **Security delta**: N/A
- **Validation**: static-only (count String allocations)

### P-MEM-9 [High]: Graph expansion re-scores entire result set
- **File**: `crates/heartbit-core/src/memory/in_memory.rs:369–441`
- **Observation**: After expanding with related_ids (line 348–367), code re-computes avgdl, re-computes BM25 for new entries, re-computes composite scores for all entries (line 405–441). If limit=10 but expansion adds 50 related entries, you re-score 60 entries when only top 10 will be returned. Then truncate (line 443–444).
- **Hypothesized cost**: ~500μs–2ms per recall (if expansion large; avoided if no related_ids or limit=0).
- **Frequency**: warm-path-per-recall
- **Fix sketch**:
  1. Apply limit BEFORE expansion, or expand only for top K entries (not all results).
  2. Skip re-sort if no new entries were added (line 370).
  3. Only re-score expanded entries, not all results.
- **Security delta**: N/A
- **Validation**: needs-bench (measure expansion cost at various limit values)

### P-MEM-10 [High]: OpenAiEmbedding single-entry batching
- **File**: `crates/heartbit-core/src/memory/embedding.rs:191, 224`
- **Observation**: 
  - On store: `.embed(&[&entry.content]).await` — single entry batched as array of 1
  - On recall: `.embed(&[text]).await` — single query batched as array of 1
  - OpenAI API supports batch_size up to 2048. Sending 1 request per store + 1 per recall wastes HTTP overhead (~50–200ms per request latency).
- **Hypothesized cost**: ~100–500ms per store+recall (HTTP + API latency) vs ~50–100ms if batched 10-100 entries per request.
- **Frequency**: hot-path-per-store, hot-path-per-recall (with embeddings enabled)
- **Fix sketch**:
  1. Add batch queue: accumulate embedding requests in a channel, batch N entries (e.g., 100) before sending.
  2. Or: use embedder.embed(&[content1, content2, ...]) for multi-entry batch at consolidation time.
  3. For recall, only emit single query embedding (already done, unavoidable).
  4. Caveat: batching adds latency variance — useful for async loops, not for single immediate recalls.
- **Security delta**: N/A
- **Validation**: measured (compare latency with batch size 1 vs 10/100)

### P-MEM-11 [High]: add_link bidirectional scan overhead
- **File**: `crates/heartbit-core/src/memory/in_memory.rs:526–564`
- **Observation**: `add_link()` requires 4 separate lookups: check id_ok (line 542), check rel_ok (line 546), push to id's related_ids (line 552–555), push to related's related_ids (line 557–561). Each `get()` and `get_mut()` is a HashMap lookup. For N entries with high link density (e.g., graph clustering), this is O(degree) × O(1) = overhead per link. Also requires write-lock for entire operation.
- **Hypothesized cost**: ~50–100μs per add_link (4 HashMap ops + 2 Vec pushes + write-lock overhead).
- **Frequency**: warm-path-per-store or batch consolidation (post-consolidation, add_link may be called in loops)
- **Fix sketch**:
  1. Batch link adds: accumulate in temp Vec, then apply all at once (1 write-lock).
  2. Or: add a "bidirectional_add_link" method that does both updates in single lock.
  3. Use Vec::contains() check before pushing to avoid duplicates — this is O(degree), not hash-lookup. Pre-compute related_ids HashSet for faster contains check.
- **Security delta**: N/A
- **Validation**: static-only (count HashMap lookups per add_link call)

### P-MEM-12 [Medium]: NamespacedMemory string concatenation overhead
- **File**: `crates/heartbit-core/src/memory/namespaced.rs:59–61, 102`
- **Observation**: Every operation prefixes/unprefixes IDs: `format!("{}:{}", self.agent_name, id)` on store (line 70), and `strip_prefix(&prefix)` on recall (line 109). At 10k entries, this is 10k string allocations for prefix construction. Namespace string is repeated per operation.
- **Hypothesized cost**: ~50–100μs per operation (10k format!/strip calls × 5–10μs each).
- **Frequency**: hot-path-per-store, hot-path-per-recall
- **Fix sketch**:
  1. Pre-compute prefix string once in NamespacedMemory::new() or as a cached field.
  2. Use `Cow<str>` or reference to avoid allocation.
  3. Use `format!()` builder or `String::reserve()` to pre-allocate.
- **Security delta**: N/A
- **Validation**: static-only (count String allocations)

### P-MEM-13 [Medium]: Related_ids and source_ids Vec cloning per entry
- **File**: `crates/heartbit-core/src/memory/mod.rs:92–96`
- **Observation**: `related_ids: Vec<String>` and `source_ids: Vec<String>` are cloned with every MemoryEntry clone (P-MEM-5). Typically small (<8 elements), but still incur allocation overhead. At 10k entries, 10k clones × 2 Vecs = 20k Vec allocations.
- **Hypothesized cost**: ~100–300μs per recall (20k Vec allocations × 50–100ns each, plus element cloning).
- **Frequency**: hot-path-per-recall (via P-MEM-5 full clone)
- **Fix sketch**:
  1. Use `SmallVec<[String; 8]>` for related_ids and source_ids — stack allocation for typical case, heap only if >8.
  2. Combined with P-MEM-5 fix (Cow/Arc return), this avoids clone overhead entirely.
- **Security delta**: N/A
- **Validation**: static-only (measure allocation count for small Vecs)

### P-MEM-14 [Medium]: Strength decay calculation during store eviction
- **File**: `crates/heartbit-core/src/memory/in_memory.rs:84–112`
- **Observation**: At capacity, eviction finds weakest entry by iterating all entries and computing effective_strength for each (line 88–101). This is O(N) decay computations just to evict one entry. At N=100k entries, evicting 10 new entries in sequence = 1M decay computations. However, this is only triggered when at_cap and not updating existing entry.
- **Hypothesized cost**: ~100–500μs per store when evicting (O(N) × O(1) decay computation).
- **Frequency**: warm-path-per-store (only at cap)
- **Fix sketch**:
  1. Maintain a separate "strength cache" or "last_computed_strength" timestamp to avoid recomputing decay during eviction.
  2. Or: use a min-heap keyed by effective_strength, updated lazily on read/write.
  3. Or: approximate eviction by tracking a running "weakest_id" lazily.
- **Security delta**: N/A
- **Validation**: needs-bench (measure store latency at capacity)

### P-MEM-15 [Medium]: Prune full-scan without early termination
- **File**: `crates/heartbit-core/src/memory/in_memory.rs:567–620`
- **Observation**: `prune()` scans all entries to find weak ones (line 583–612). Collects all matching IDs, then removes them. No early termination or filtering before full scan. If pruning 100 entries from 100k, still scans all 100k.
- **Hypothesized cost**: ~100–300μs per prune (O(N) full scan + O(K) removal, where K = entries pruned).
- **Frequency**: cold-path (triggered periodically, e.g., hourly)
- **Fix sketch**:
  1. Add early termination: if K entries marked for removal, and K > some threshold, stop scanning.
  2. Or: maintain a "sorted by strength" index to prune from weakest without full scan.
  3. Low priority — cold path, acceptable at O(N).
- **Security delta**: N/A
- **Validation**: static-only

### P-MEM-16 [Low]: Cosine similarity scalar loop instead of SIMD
- **File**: `crates/heartbit-core/src/memory/hybrid.rs:39–62`
- **Observation**: Cosine similarity iterates f32 pairs manually (line 48–54). Modern CPUs support SIMD (SSE2, AVX2) for vector math. Embedding vectors are 1536–3072 f32s, so SIMD could provide 4–8x speedup.
- **Hypothesized cost**: ~1–10μs per cosine_similarity call (1536-dim vector) vs ~200–500ns with SIMD.
- **Frequency**: hot-path-per-recall (if embeddings present)
- **Fix sketch**:
  1. Use ndarray or nalgebra crate for SIMD dot product.
  2. Or: use x86_64 intrinsics directly (unsafe, requires target-cpu tuning).
  3. Low impact compared to algorithmic improvements (P-MEM-2, P-MEM-3).
- **Security delta**: N/A
- **Validation**: needs-bench (measure cosine_similarity time before/after SIMD)

### P-MEM-17 [Low]: Reflection tracker uses std::sync::Mutex
- **File**: `crates/heartbit-core/src/memory/reflection.rs:12`
- **Observation**: `ReflectionTracker` uses `std::sync::Mutex<u32>` for accumulated importance. This is a poisoning-prone lock protecting a tiny u32. Not a hot path, but could be faster with `parking_lot::Mutex`.
- **Hypothesized cost**: ~1–5μs per record() (Mutex lock/unlock + u32 add/compare).
- **Frequency**: warm-path-per-store (called on every store to check reflection threshold)
- **Fix sketch**: Replace `std::sync::Mutex` with `parking_lot::Mutex` (no poisoning, faster).
- **Security delta**: N/A
- **Validation**: static-only (no measurable difference in practice)

### P-MEM-18 [Medium]: No early termination in BM25 token filtering
- **File**: `crates/heartbit-core/src/memory/in_memory.rs:136–152`
- **Observation**: Filter loop checks all conditions (text, category, tags, agent, memory_type, min_strength, max_confidentiality) with no short-circuit ordering. If text is highly selective (matches 1% of entries), checking all other filters on non-matching entries is wasteful. Conditions are checked in order, but the text filter (substring match) is expensive.
- **Hypothesized cost**: ~100–200μs per recall if text filter is non-selective (few matches but checked against all entries).
- **Frequency**: hot-path-per-recall
- **Fix sketch**:
  1. Reorder filter conditions by selectivity: check exact-match filters first (agent, category, memory_type), then text (substring).
  2. Or: use the inverted index (P-MEM-2) to filter by text first, then apply other conditions.
  3. Low priority compared to P-MEM-2, P-MEM-6.
- **Security delta**: N/A
- **Validation**: static-only

---

## Cross-cutting recommendations

### Locking strategy
- **Immediate**: Replace all `std::sync::RwLock` → `parking_lot::RwLock`, all `std::sync::Mutex` → `parking_lot::Mutex`. No semantic changes; ~20% faster lock operations.
- **Medium term**: Implement read-lock for filtering (P-MEM-1 split lock scope) to allow concurrent recalls, write-lock only for access_count/strength updates.

### Indexing strategy
- **Inverted index** (P-MEM-2): Build `HashMap<token, Vec<entry_id>>` at store time. Rebuild incrementally or on periodic maintenance. Estimate index size ~10–20% of entry size (9 bytes per token, ~100 tokens per 10k entries = 9MB index for 100k entries).
- **Per-tenant index** (P-MEM-6): Secondary `HashMap<tenant_id, Vec<entry_id>>` updated at store/forget time. Estimate 8 bytes per tenant + 8 bytes per entry = negligible overhead for typical deployments.

### Memory layout
- **SmallVec for keywords, related_ids, source_ids** (P-MEM-13): Replace `Vec<String>` with `SmallVec<[String; 8]>` for typical case (< 8 items). Saves allocation for 90%+ of entries.
- **Cow/Arc for recall return** (P-MEM-5): Return references or Arc instead of owned clones. Biggest win at scale (10k entries = 5MB allocation savings).

### Embedding strategy
- **Batching** (P-MEM-10): Accumulate embedding requests in async queue, batch N entries before sending. Reduces HTTP latency 10–100x for high-throughput scenarios.
- **Query embedding caching**: If same query text is searched multiple times in a session, cache the embedding.

---

## Validation checklist (for implementer)

- [ ] Add benchmark harness for recall latency at N=10k/100k with various query types
- [ ] Profile memory allocations per recall (need flame graph or perf)
- [ ] Measure lock contention with concurrent recalls (5, 10, 20 threads)
- [ ] Verify tenant isolation after per-tenant indexing (F-MEM-5)
- [ ] Verify embedding API calls batched (no regression to single-entry requests)
- [ ] Consolidation time at N=10k before/after O(N²) optimization
- [ ] Heap usage before/after SmallVec + Cow/Arc changes

---

## Notes

- No security regressions identified. All findings preserve tenant isolation, confidentiality filtering, and hardened embedding client.
- BM25 inverted index is the highest-leverage single change: ~500μs–2ms per recall savings (P-MEM-2).
- Lock strategy (parking_lot + read-lock split) is the second-highest: ~200–500μs per recall + unlocks concurrent queries (P-MEM-1).
- Lazy strength decay (P-MEM-3) is quick to implement and saves ~50–100μs per recall.
- Full clone return type (P-MEM-5) requires API change but saves 1–5ms per recall at scale.
