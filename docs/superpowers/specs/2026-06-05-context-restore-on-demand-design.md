# Context Restore-on-Demand — Design Spec

**Date:** 2026-06-05
**Status:** Approved (brainstorming), pending implementation plan
**Source:** Leverage #1 of the context-management deep research (`wf_24398882-8ae`)

## 1. Problem

An agentic loop is a recurrence: every turn re-sends the prior transcript. As the
window fills with tool outputs it degrades on three axes at once — quality
(accuracy falls well before the token limit), cost (input billing grows
~quadratically), and latency (stacked O(n²) prefill). The standard mitigation is
to drop/truncate old content. heartbit already has the machinery (session
pruner, auto-compaction), but it is **lossy and irreversible**: once an old tool
result is truncated to `[pruned: N bytes]`, the model can never get it back, so
the harness must prune timidly.

**Reversibility is what licenses aggression.** If dropped content can be restored
on demand, the harness can prune/compact hard without permanent loss. This spec
designs that restore path.

## 2. Goals / Non-Goals

### Goals
- Index every tool output, by `tool_call_id`, into a per-run store at the moment
  it is produced.
- `fetch_full_output(ref)` — restore the exact untruncated content of a specific
  past tool result.
- `recall_context(query)` — semantically find dropped/old outputs by meaning when
  the model doesn't remember the exact ref.
- Make the pruner's truncation marker **actionable**: it names the ref to fetch.
- Make restore *discoverable*: a system-prompt hint (gated on the feature flag).

### Non-Goals (deliberately deferred)
- Changing pruner/compactor **defaults** (e.g. `Unlimited` → sliding window,
  pruner-on-by-default). That is leverage #2/#3 and carries the prompt-cache
  invalidation tension; bundling it would mix concerns.
- Append-only / cache-aware compaction (leverage #2).
- Cross-run / persistent recall (that is the Memory system's job; this store is
  per-run).

## 3. Background — what already exists (grounded)

- **Pruner** (`agent/pruner.rs`): `prune_old_tool_results(messages, config)` returns
  a *new* message list with old `ToolResult` blocks truncated to ~200B + a
  `[pruned: N bytes]` marker. It is applied **per-request** (`runner.rs:607-613`)
  — the stored conversation keeps full content; only the per-LLM-call request is
  shrunk. Every `ToolResult` block carries a `tool_use_id`.
- **Compaction** (`runner.rs`, reactive on `ContextOverflow`): destructive —
  `inject_summary` replaces old messages with a summary.
- **Full output is already in hand** in the tool loop (`runner.rs` ~2444): the
  runner already captures the untruncated `output.content` alongside the
  `tool_call_id` (for audit + `ToolCallRecord`).
- **Retrieval is already built and wired** in `InMemoryStore::recall`
  (`memory/in_memory.rs:457`): BM25 (`memory/bm25.rs`) + vector → `rrf_fuse`
  (`memory/hybrid.rs:16`). The `EmbeddingProvider` trait (`memory/embedding.rs`)
  degrades cleanly: `NoopEmbedding` / `dimension()==0` → BM25-only (default); a
  real embedder → hybrid. `InMemoryStore` keeps `entries: HashMap<id, MemoryEntry>`
  and a `max_entries` LRU cap.

**Implication:** the "semantic recall" half is mostly *reuse*, not new retrieval
code. The exact-fetch half needs one small additive read on `InMemoryStore`.

## 4. Design

### 4.1 Components (each a unit with one purpose)

| Unit | File | Responsibility | Depends on |
|---|---|---|---|
| `ContextRecallStore` | `agent/context_recall.rs` (new) | per-run store of tool outputs by `tool_call_id`: `index` / `get` / `recall` | wraps a reused `InMemoryStore` |
| `InMemoryStore::get(id)` | `memory/in_memory.rs` (+~3 lines) | exact O(1) by-id read of a `MemoryEntry` | existing `entries` map |
| `FetchFullOutputTool` | `tool/builtins/fetch_full_output.rs` (new) | `fetch_full_output(ref)` → exact untruncated content | `Arc<ContextRecallStore>` |
| `RecallContextTool` | `tool/builtins/recall_context.rs` (new) | `recall_context(query, limit?)` → ranked `{ref, tool_name, snippet}` | `Arc<ContextRecallStore>` |
| indexing hook | `agent/runner.rs` tool loop | `store.index(id, name, full_content)` when a result is produced | store |
| ref-marker | `agent/pruner.rs` | truncation marker carries the `tool_use_id` | — |
| wiring | `agent/builder.rs` + `tool/builtins/mod.rs` (`BuiltinToolsConfig`) | create/share `Arc<ContextRecallStore>`, opt-in flag, gate tool registration + prompt hint | — |

### 4.2 `ContextRecallStore` API (sketch)

```rust
pub struct ContextRecallStore { inner: InMemoryStore } // bounded via with_max_entries

impl ContextRecallStore {
    pub fn new() -> Self;                       // default cap (~200 entries)
    pub fn with_capacity(max_entries: usize) -> Self;
    async fn index(&self, tool_call_id: &str, tool_name: &str, content: &str);
    async fn get(&self, tool_call_id: &str) -> Option<String>;   // exact
    async fn recall(&self, query: &str, limit: usize) -> Vec<RecallHit>; // BM25(/+vector)
}
pub struct RecallHit { pub r#ref: String, pub tool_name: String, pub snippet: String }
```

- `index` writes a `MemoryEntry { id = tool_call_id, content, tags = [tool_name] }`.
  Embedding is automatic & gated by the configured `EmbeddingProvider`
  (BM25-only by default).
- `get` calls the new `InMemoryStore::get(id)` and returns `entry.content`.
- `recall` delegates to `InMemoryStore::recall`, then projects hits to
  `{ref, tool_name, snippet}` with a **generous snippet** (~200–300 chars, head).

### 4.3 Tool surfaces

- `fetch_full_output(ref: string)` → the exact stored content, or a clean
  `ToolOutput::error("no stored output for ref X — it may have been evicted")`.
- `recall_context(query: string, limit?: int=5)` → a compact ranked list of
  `{ref, tool_name, snippet}`. **Two-step by design:** recall to *find*, fetch to
  *restore* — so the recall result never re-bloats the window with full bodies.
  Generous snippets mean the snippet often answers the question and no fetch is
  needed.

### 4.4 Data flow

```
tool runs → index(id, name, full_content)            [ALWAYS, at production time]
   … turns pass …
pruner truncates old result in the per-call request →
   "<head>…<tail> [pruned 4.2KB — fetch_full_output(\"tc_abc\") to restore]"
model sees marker →
   ├─ knows the ref → fetch_full_output("tc_abc") → exact full content
   └─ forgot it     → recall_context("the failing test output") → ranked {ref,…}
tool returns content as a NEW tool result → re-enters the conversation
   (and is itself re-indexed → restorable again)
```

### 4.5 Reversibility property

Because indexing happens at **production** time (not at drop time), content hidden
by *either* the pruner (per-request) *or* the compactor (destructive) is
restorable. The store is the single source of truth for "everything the agent ever
saw" this run.

## 5. Enablement & discoverability

- Opt-in via `AgentRunnerBuilder::context_recall(bool)`, **off by default**.
- **Everything is gated on the flag — off means zero overhead.** When off: the
  store is not created, the indexing hook is skipped, and the two tools are not
  registered. There is no cost (no extra storage, no indexing work, no
  tool-definition tokens) unless the feature is explicitly enabled.
- **Why gate tool registration specifically:** registering `fetch_full_output` /
  `recall_context` while nothing prunes would hand the model two tools that can
  never return anything *and* cost tool-definition tokens every turn —
  self-defeating for a context-saving feature.
- **Indexing scope:** when on, *every* tool result is indexed at production time,
  including error outputs (an error result is just as worth restoring/searching).
- When on, append a one-line system-prompt hint: *"Old tool outputs may be
  truncated with a `fetch_full_output(...)` marker — call it to restore what you
  need; use `recall_context(query)` to find older content by meaning."* The whole
  value hinges on the model actually calling these on a marker; the hint is the
  behavioral linchpin.
- Most meaningful turned on **alongside the pruner**.

## 6. Error handling & bounds

- Unknown/evicted ref → clean tool error, never a hard run failure.
- No embedder → BM25-only automatically (`dimension()==0` guard).
- Store bounded by `with_max_entries` (existing LRU eviction) → cannot leak;
  recent/relevant content survives, evicted refs degrade gracefully.

## 7. Cache note (in-scope honesty)

The pruner is per-request, so it *already* busts the prompt-cache prefix each turn
it prunes. This design does **not** worsen that: restores are *appended* as new
tool results (cache-friendly). The append-only / cache-aware compaction win is
leverage #2, out of scope here.

## 8. Testing strategy (TDD — slices)

1. `InMemoryStore::get` — by-id hit returns the entry; miss → `None`.
2. `ContextRecallStore` — `index` then `get` roundtrips the exact content;
   `get(unknown)` → `None`; `recall` ranks a semantically/lexically matching
   output above a noise one (BM25-only path, no embedder); eviction drops the
   oldest past the cap.
3. `FetchFullOutputTool` — `definition` schema; `execute(known)` → content;
   `execute(unknown)` → error output (not a panic).
4. `RecallContextTool` — `execute(query)` → ranked `{ref,…}`; empty store → empty;
   snippet is capped.
5. `pruner` — a truncated `ToolResult` marker contains its `tool_use_id` and the
   literal `fetch_full_output`.
6. runner integration — after a (mock) tool runs with `context_recall` on, the
   store holds the full output by id; with it off, the tools are not registered.
7. end-to-end — produce a large output → prune → marker carries the ref →
   `fetch_full_output(ref)` returns the original full content.

## 9. Sequencing (strategic)

**This lands before any aggressive compaction work (leverage #2).** Reversibility
is the precondition that makes turning compaction up *safe*. Build order: #1 (this)
→ then #2.

## 10. Open questions / risks

- **Behavioral reliance:** does the model reliably call fetch/recall on a marker?
  Mitigated by the self-documenting marker + prompt hint; worth watching in live
  use. If under-used, consider auto-restoring on the next turn instead of relying
  on the model.
- **Default cap value (~200):** a starting guess; revisit if evictions bite real
  sessions.
- **Snippet length:** ~200–300 chars head; tune from live use.
