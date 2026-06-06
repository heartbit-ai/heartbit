# Context Restore-on-Demand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the session pruner/compactor reversible — index every tool output by `tool_call_id` into a per-run store, and give the agent `fetch_full_output(ref)` (exact) + `recall_context(query)` (semantic) tools to restore dropped content on demand.

**Architecture:** A per-run `ContextRecallStore` wraps a reused `InMemoryStore` (which already does BM25+vector→RRF recall). The runner indexes each tool result as a `MemoryEntry` keyed by `tool_call_id` when it is produced. Two new builtin tools read that store. The pruner's truncation marker carries the `tool_use_id` so a pruned result names its own ref. Everything is gated on the presence of a shared `Arc<ContextRecallStore>` — off means zero overhead.

**Tech Stack:** Rust, tokio (async tools via `Pin<Box<dyn Future>>`), `parking_lot::RwLock`, `serde_json`, `chrono`. Reuses `crate::memory::{Memory, InMemoryStore, MemoryEntry, MemoryQuery}` and `crate::auth::tenant::TenantScope`.

**Quality gate (run before every commit):**
```bash
cargo fmt --all -- --check && cargo clippy --workspace --exclude mini-crm --all-targets -- -D warnings && cargo test --workspace --exclude mini-crm
```

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `crates/heartbit-core/src/memory/in_memory.rs` | Modify | add `InMemoryStore::get(id)` (exact by-id read) |
| `crates/heartbit-core/src/agent/context_recall.rs` | Create | `ContextRecallStore` — `index` / `get` / `recall` over a reused `InMemoryStore` |
| `crates/heartbit-core/src/agent/mod.rs` | Modify | `pub mod context_recall;` |
| `crates/heartbit-core/src/lib.rs` | Modify | re-export `ContextRecallStore` |
| `crates/heartbit-core/src/tool/builtins/fetch_full_output.rs` | Create | `FetchFullOutputTool` (`fetch_full_output(ref)`) |
| `crates/heartbit-core/src/tool/builtins/recall_context.rs` | Create | `RecallContextTool` (`recall_context(query, limit?)`) |
| `crates/heartbit-core/src/tool/builtins/mod.rs` | Modify | module decls + `BuiltinToolsConfig.context_recall_store` + register tools when present |
| `crates/heartbit-core/src/agent/pruner.rs` | Modify | put `tool_use_id` in the truncation marker |
| `crates/heartbit-core/src/agent/builder.rs` | Modify | `context_recall_store` field + setter + build() transfer + system-prompt hint |
| `crates/heartbit-core/src/agent/runner.rs` | Modify | `context_recall_store` field + indexing hook in the tool loop |

---

## Task 1: `InMemoryStore::get` — exact by-id read

**Files:**
- Modify: `crates/heartbit-core/src/memory/in_memory.rs`
- Test: same file, `#[cfg(test)] mod tests`

- [ ] **Step 1: Write the failing test** (add inside the existing `#[cfg(test)] mod tests` block):

```rust
#[tokio::test]
async fn get_returns_entry_by_id_or_none() {
    use crate::auth::tenant::TenantScope;
    let store = InMemoryStore::new();
    let scope = TenantScope::default();
    let mut entry = sample_entry("e1", "hello world"); // existing test helper
    entry.id = "e1".into();
    store.store(&scope, entry).await.unwrap();

    let got = store.get("e1").expect("entry e1 should exist");
    assert_eq!(got.content, "hello world");
    assert!(store.get("missing").is_none());
}
```

> If no `sample_entry` helper exists in this test module, build the entry with the struct literal from Task 2 Step 3 instead (set `id: "e1"`, `content: "hello world"`).

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p heartbit-core --lib memory::in_memory::tests::get_returns_entry_by_id_or_none`
Expected: FAIL — `no method named 'get' found for struct 'InMemoryStore'`.

- [ ] **Step 3: Write minimal implementation** (add to the existing `impl InMemoryStore` block, near `with_max_entries`):

```rust
    /// Exact lookup of a stored entry by id (no recall scoring, no reinforcement).
    pub fn get(&self, id: &str) -> Option<MemoryEntry> {
        self.entries.read().get(id).cloned()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p heartbit-core --lib memory::in_memory::tests::get_returns_entry_by_id_or_none`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all -- --check && cargo clippy -p heartbit-core --all-targets -- -D warnings
git add crates/heartbit-core/src/memory/in_memory.rs
git commit -m "feat(memory): InMemoryStore::get for exact by-id lookup"
```

---

## Task 2: `ContextRecallStore` — index + get

**Files:**
- Create: `crates/heartbit-core/src/agent/context_recall.rs`
- Modify: `crates/heartbit-core/src/agent/mod.rs` (add `pub mod context_recall;`)
- Test: in `context_recall.rs`

- [ ] **Step 1: Register the module** — add to `crates/heartbit-core/src/agent/mod.rs` next to the other `pub mod` lines (e.g. after `pub mod context;`):

```rust
pub mod context_recall;
```

- [ ] **Step 2: Write the failing test** — create `crates/heartbit-core/src/agent/context_recall.rs` with ONLY the test first (plus imports):

```rust
//! Per-run "restore-on-demand" store: indexes every tool output by
//! `tool_call_id` so a pruned/compacted result can be restored exactly
//! (`get`) or found semantically (`recall`). Reuses `InMemoryStore`'s
//! BM25(+vector)→RRF retrieval.

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn index_then_get_roundtrips_exact_content() {
        let store = ContextRecallStore::new();
        store.index("tc_1", "bash", "the full untruncated output").await;
        assert_eq!(
            store.get("tc_1").await.as_deref(),
            Some("the full untruncated output")
        );
        assert_eq!(store.get("nope").await, None);
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p heartbit-core --lib agent::context_recall::tests::index_then_get_roundtrips_exact_content`
Expected: FAIL — `cannot find type 'ContextRecallStore'`.

- [ ] **Step 4: Write minimal implementation** — add above the `#[cfg(test)]` block in `context_recall.rs`:

```rust
use crate::auth::tenant::TenantScope;
use crate::memory::in_memory::InMemoryStore;
use crate::memory::{Confidentiality, Memory, MemoryEntry, MemoryType};

/// Default cap on stored tool outputs (bounded so the store can't leak).
const DEFAULT_MAX_ENTRIES: usize = 256;

/// A per-run store of tool outputs, keyed by `tool_call_id`.
pub struct ContextRecallStore {
    inner: InMemoryStore,
    scope: TenantScope,
}

impl Default for ContextRecallStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextRecallStore {
    /// Create a bounded store with the default capacity.
    pub fn new() -> Self {
        Self {
            inner: InMemoryStore::new().with_max_entries(DEFAULT_MAX_ENTRIES),
            scope: TenantScope::default(),
        }
    }

    /// Index a tool result so it can later be restored by `id` or found by query.
    /// Best-effort: a store error is swallowed (recall is a convenience, not a
    /// correctness dependency).
    pub async fn index(&self, tool_call_id: &str, tool_name: &str, content: &str) {
        let entry = make_entry(tool_call_id, tool_name, content);
        let _ = self.inner.store(&self.scope, entry).await;
    }

    /// Exact restore of a stored tool output by its `tool_call_id`.
    pub async fn get(&self, tool_call_id: &str) -> Option<String> {
        self.inner.get(tool_call_id).map(|e| e.content)
    }
}

/// Build a `MemoryEntry` for a tool output (defaults for the unused fields).
fn make_entry(id: &str, tool_name: &str, content: &str) -> MemoryEntry {
    let now = chrono::Utc::now();
    MemoryEntry {
        id: id.to_string(),
        agent: "context_recall".into(),
        content: content.to_string(),
        category: "tool_output".into(),
        tags: vec![tool_name.to_string()],
        created_at: now,
        last_accessed: now,
        access_count: 0,
        importance: 5,
        memory_type: MemoryType::Episodic,
        keywords: Vec::new(),
        summary: None,
        strength: 1.0,
        related_ids: Vec::new(),
        source_ids: Vec::new(),
        embedding: None,
        confidentiality: Confidentiality::Public,
        author_user_id: None,
        author_tenant_id: None,
    }
}
```

> If the compiler reports different variant names, use the actual ones (`MemoryType::Episodic`, `Confidentiality::Public` are the documented defaults). If `TenantScope` is re-exported at `crate::TenantScope`, either path works.

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p heartbit-core --lib agent::context_recall::tests::index_then_get_roundtrips_exact_content`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cargo fmt --all -- --check && cargo clippy -p heartbit-core --all-targets -- -D warnings
git add crates/heartbit-core/src/agent/context_recall.rs crates/heartbit-core/src/agent/mod.rs
git commit -m "feat(agent): ContextRecallStore index+get (reuses InMemoryStore)"
```

---

## Task 3: `ContextRecallStore::recall` — semantic find (two-step)

**Files:**
- Modify: `crates/heartbit-core/src/agent/context_recall.rs`
- Test: same file

- [ ] **Step 1: Write the failing test** — add to the `mod tests` block:

```rust
    #[tokio::test]
    async fn recall_ranks_a_matching_output_above_noise_and_caps_snippet() {
        let store = ContextRecallStore::new();
        store
            .index("tc_match", "bash", "cargo test failed: assertion error in parser module")
            .await;
        store
            .index("tc_noise", "read", "the quick brown fox jumps over the lazy dog")
            .await;

        let hits = store.recall("test failure parser", 5).await;
        assert!(!hits.is_empty(), "expected at least one hit");
        assert_eq!(hits[0].r#ref, "tc_match", "the matching output must rank first");
        assert_eq!(hits[0].tool_name, "bash");
        assert!(hits[0].snippet.chars().count() <= SNIPPET_CHARS);
    }

    #[tokio::test]
    async fn recall_on_empty_store_is_empty() {
        let store = ContextRecallStore::new();
        assert!(store.recall("anything", 5).await.is_empty());
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p heartbit-core --lib agent::context_recall::tests::recall_`
Expected: FAIL — `no method named 'recall'` and `cannot find value 'SNIPPET_CHARS'`.

- [ ] **Step 3: Write minimal implementation** — add the `RecallHit` struct + const near the top of `context_recall.rs`, and the `recall` method to the `impl`:

```rust
use crate::memory::MemoryQuery;

/// Max characters of head-content returned per recall hit (generous, so the
/// snippet often answers the question without a follow-up fetch).
const SNIPPET_CHARS: usize = 280;

/// One ranked match from `recall`: the ref to fetch, which tool produced it,
/// and a head snippet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecallHit {
    pub r#ref: String,
    pub tool_name: String,
    pub snippet: String,
}
```

```rust
    /// Semantically find stored tool outputs by `query` (BM25, or BM25+vector
    /// when an embedder is configured). Returns ranked refs + head snippets;
    /// the caller restores the full body via `get`/`fetch_full_output`.
    pub async fn recall(&self, query: &str, limit: usize) -> Vec<RecallHit> {
        let q = MemoryQuery {
            text: Some(query.to_string()),
            limit,
            reinforce: false,
            ..Default::default()
        };
        let entries = self.inner.recall(&self.scope, q).await.unwrap_or_default();
        entries
            .into_iter()
            .map(|e| RecallHit {
                r#ref: e.id,
                tool_name: e.tags.first().cloned().unwrap_or_default(),
                snippet: e.content.chars().take(SNIPPET_CHARS).collect(),
            })
            .collect()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p heartbit-core --lib agent::context_recall::tests::recall_`
Expected: PASS (both tests).

> If `recall` ranks differently than asserted (RRF tie-break), keep the lexical-overlap query strong enough that `tc_match` wins; the noise entry shares no query terms, so BM25 should rank it last or exclude it.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all -- --check && cargo clippy -p heartbit-core --all-targets -- -D warnings
git add crates/heartbit-core/src/agent/context_recall.rs
git commit -m "feat(agent): ContextRecallStore::recall (ranked refs+snippets, two-step)"
```

---

## Task 4: re-export `ContextRecallStore`

**Files:**
- Modify: `crates/heartbit-core/src/lib.rs`

- [ ] **Step 1: Add the re-export** — next to the other `agent` re-exports (e.g. near `pub use agent::...`):

```rust
pub use agent::context_recall::{ContextRecallStore, RecallHit};
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build -p heartbit-core`
Expected: builds clean.

- [ ] **Step 3: Commit**

```bash
git add crates/heartbit-core/src/lib.rs
git commit -m "feat(agent): re-export ContextRecallStore"
```

---

## Task 5: `FetchFullOutputTool`

**Files:**
- Create: `crates/heartbit-core/src/tool/builtins/fetch_full_output.rs`
- Modify: `crates/heartbit-core/src/tool/builtins/mod.rs` (add `mod fetch_full_output;`)
- Test: in `fetch_full_output.rs`

- [ ] **Step 1: Register the module** — add to `crates/heartbit-core/src/tool/builtins/mod.rs` near the other `mod` lines:

```rust
mod fetch_full_output;
```

- [ ] **Step 2: Write the failing test** — create `fetch_full_output.rs` with imports + the test first:

```rust
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::json;

use crate::agent::context_recall::ContextRecallStore;
use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fetches_known_ref_and_errors_on_unknown() {
        let store = Arc::new(ContextRecallStore::new());
        store.index("tc_1", "bash", "FULL OUTPUT BODY").await;
        let tool = FetchFullOutputTool { store: store.clone() };
        let ctx = crate::ExecutionContext::default();

        let ok = tool
            .execute(&ctx, json!({ "ref": "tc_1" }))
            .await
            .unwrap();
        assert_eq!(ok.content, "FULL OUTPUT BODY");
        assert!(!ok.is_error);

        let miss = tool
            .execute(&ctx, json!({ "ref": "nope" }))
            .await
            .unwrap();
        assert!(miss.is_error);
        assert!(miss.content.contains("nope"));
    }

    #[test]
    fn definition_declares_ref_param() {
        let tool = FetchFullOutputTool { store: Arc::new(ContextRecallStore::new()) };
        let def = tool.definition();
        assert_eq!(def.name, "fetch_full_output");
        assert!(def.input_schema["properties"].get("ref").is_some());
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p heartbit-core --lib tool::builtins::fetch_full_output`
Expected: FAIL — `cannot find type 'FetchFullOutputTool'`.

- [ ] **Step 4: Write minimal implementation** — add above the `#[cfg(test)]` block:

```rust
/// Restores the exact untruncated content of a past tool result by its ref
/// (the `tool_call_id`), e.g. after the pruner truncated it.
pub(crate) struct FetchFullOutputTool {
    pub(crate) store: Arc<ContextRecallStore>,
}

impl Tool for FetchFullOutputTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "fetch_full_output".into(),
            description: "Restore the full, untruncated content of an earlier tool result by \
                its ref. Use when an old tool output shows a '[pruned: … id=<ref>]' marker and \
                you need the complete content back."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "The ref (tool_call_id) shown in the pruned marker."
                    }
                },
                "required": ["ref"]
            }),
        }
    }

    fn execute(
        &self,
        _ctx: &crate::ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let r = input
                .get("ref")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("ref is required".into()))?;
            match self.store.get(r).await {
                Some(content) => Ok(ToolOutput::success(content)),
                None => Ok(ToolOutput::error(format!(
                    "no stored output for ref '{r}' — it may have been evicted"
                ))),
            }
        })
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p heartbit-core --lib tool::builtins::fetch_full_output`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
cargo fmt --all -- --check && cargo clippy -p heartbit-core --all-targets -- -D warnings
git add crates/heartbit-core/src/tool/builtins/fetch_full_output.rs crates/heartbit-core/src/tool/builtins/mod.rs
git commit -m "feat(builtins): fetch_full_output tool (exact restore by ref)"
```

---

## Task 6: `RecallContextTool`

**Files:**
- Create: `crates/heartbit-core/src/tool/builtins/recall_context.rs`
- Modify: `crates/heartbit-core/src/tool/builtins/mod.rs` (add `mod recall_context;`)
- Test: in `recall_context.rs`

- [ ] **Step 1: Register the module** — add to `crates/heartbit-core/src/tool/builtins/mod.rs`:

```rust
mod recall_context;
```

- [ ] **Step 2: Write the failing test** — create `recall_context.rs`:

```rust
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::json;

use crate::agent::context_recall::ContextRecallStore;
use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn returns_ranked_refs_for_a_query() {
        let store = Arc::new(ContextRecallStore::new());
        store.index("tc_match", "bash", "cargo test failed: parser assertion error").await;
        store.index("tc_noise", "read", "lorem ipsum dolor sit amet").await;
        let tool = RecallContextTool { store };
        let ctx = crate::ExecutionContext::default();

        let out = tool
            .execute(&ctx, json!({ "query": "test failure parser" }))
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("tc_match"), "ranked refs must include the match:\n{}", out.content);
        assert!(out.content.contains("fetch_full_output"), "must tell the model how to restore");
    }

    #[test]
    fn definition_declares_query_param() {
        let tool = RecallContextTool { store: Arc::new(ContextRecallStore::new()) };
        let def = tool.definition();
        assert_eq!(def.name, "recall_context");
        assert!(def.input_schema["properties"].get("query").is_some());
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p heartbit-core --lib tool::builtins::recall_context`
Expected: FAIL — `cannot find type 'RecallContextTool'`.

- [ ] **Step 4: Write minimal implementation**:

```rust
/// Semantically find earlier tool outputs by meaning (when you don't remember
/// the exact ref). Returns ranked refs + snippets; restore the full body with
/// fetch_full_output(ref).
pub(crate) struct RecallContextTool {
    pub(crate) store: Arc<ContextRecallStore>,
}

impl Tool for RecallContextTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "recall_context".into(),
            description: "Find earlier tool outputs by meaning when you don't recall the exact \
                ref. Returns ranked {ref, tool, snippet}; call fetch_full_output(ref) to restore \
                the full body of the one you want."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What to search for, in words." },
                    "limit": { "type": "integer", "description": "Max results (default 5)." }
                },
                "required": ["query"]
            }),
        }
    }

    fn execute(
        &self,
        _ctx: &crate::ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let query = input
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("query is required".into()))?;
            let limit = input
                .get("limit")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(5);

            let hits = self.store.recall(query, limit).await;
            if hits.is_empty() {
                return Ok(ToolOutput::success("No matching earlier outputs.".to_string()));
            }
            let mut out = format!("{} match(es) — call fetch_full_output(ref) to restore:\n", hits.len());
            for (i, h) in hits.iter().enumerate() {
                out.push_str(&format!(
                    "[{}] ref={} tool={} — {}\n",
                    i + 1,
                    h.r#ref,
                    h.tool_name,
                    h.snippet.replace('\n', " ")
                ));
            }
            Ok(ToolOutput::success(out))
        })
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p heartbit-core --lib tool::builtins::recall_context`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
cargo fmt --all -- --check && cargo clippy -p heartbit-core --all-targets -- -D warnings
git add crates/heartbit-core/src/tool/builtins/recall_context.rs crates/heartbit-core/src/tool/builtins/mod.rs
git commit -m "feat(builtins): recall_context tool (semantic find, two-step)"
```

---

## Task 7: register both tools when the store is present

**Files:**
- Modify: `crates/heartbit-core/src/tool/builtins/mod.rs`
- Test: same file

- [ ] **Step 1: Write the failing test** — add to the `#[cfg(test)] mod tests` block in `mod.rs`:

```rust
    #[test]
    fn context_recall_tools_registered_only_when_store_present() {
        use std::sync::Arc;
        // Off: no store → tools absent.
        let off = builtin_tools(BuiltinToolsConfig::default());
        assert!(!off.iter().any(|t| t.definition().name == "fetch_full_output"));
        assert!(!off.iter().any(|t| t.definition().name == "recall_context"));

        // On: store present → both tools registered.
        let cfg = BuiltinToolsConfig {
            context_recall_store: Some(Arc::new(crate::agent::context_recall::ContextRecallStore::new())),
            ..Default::default()
        };
        let on = builtin_tools(cfg);
        assert!(on.iter().any(|t| t.definition().name == "fetch_full_output"));
        assert!(on.iter().any(|t| t.definition().name == "recall_context"));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p heartbit-core --lib tool::builtins::tests::context_recall_tools_registered_only_when_store_present`
Expected: FAIL — `BuiltinToolsConfig` has no field `context_recall_store`.

- [ ] **Step 3a: Add the config field** — in `BuiltinToolsConfig` struct (after `skill_dirs`):

```rust
    /// When present, registers the `fetch_full_output` / `recall_context` tools
    /// and is shared with the runner for indexing. `None` = feature off (zero
    /// overhead: tools not registered, no indexing).
    pub context_recall_store: Option<Arc<crate::agent::context_recall::ContextRecallStore>>,
```

- [ ] **Step 3b: Default it** — in the `impl Default for BuiltinToolsConfig` block (after `skill_dirs: Vec::new(),`):

```rust
            context_recall_store: None,
```

- [ ] **Step 3c: Register the tools** — in `builtin_tools`, just before the allowlist `retain` (so allowlist filtering still applies):

```rust
    if let Some(store) = &config.context_recall_store {
        tools.push(Arc::new(fetch_full_output::FetchFullOutputTool { store: store.clone() }));
        tools.push(Arc::new(recall_context::RecallContextTool { store: store.clone() }));
    }
```

> The tool structs are `pub(crate)` with `pub(crate)` `store` fields (Task 5/6), so this construction compiles from `mod.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p heartbit-core --lib tool::builtins::tests::context_recall_tools_registered_only_when_store_present`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all -- --check && cargo clippy -p heartbit-core --all-targets -- -D warnings
git add crates/heartbit-core/src/tool/builtins/mod.rs
git commit -m "feat(builtins): register restore tools when a ContextRecallStore is present"
```

---

## Task 8: pruner marker carries the `tool_use_id`

**Files:**
- Modify: `crates/heartbit-core/src/agent/pruner.rs`
- Test: same file

- [ ] **Step 1: Write the failing test** — add to the `#[cfg(test)] mod tests` block in `pruner.rs`:

```rust
    #[test]
    fn pruned_marker_includes_the_tool_use_id() {
        let big = "x".repeat(5000);
        let messages = vec![
            Message::user("task"),
            Message { role: Role::User, content: vec![ContentBlock::ToolResult {
                tool_use_id: "tc_abc".into(),
                content: big,
                is_error: false,
            }]},
            Message::user("recent 1"),
            Message::assistant("recent 2"),
            Message::user("recent 3"),
            Message::assistant("recent 4"),
        ];
        let (out, stats) = prune_old_tool_results(&messages, &SessionPruneConfig::default());
        assert!(stats.did_prune());
        let pruned_text: String = out
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|b| match b {
                ContentBlock::ToolResult { content, .. } => Some(content.clone()),
                _ => None,
            })
            .collect();
        assert!(pruned_text.contains("tc_abc"), "marker must name the ref: {pruned_text}");
        assert!(pruned_text.contains("pruned"), "marker should still say pruned");
    }
```

> Match the message-construction helpers actually used elsewhere in this test module (e.g. `Message::user`, `Message::assistant`). If the tail-keep count differs, add enough recent messages that the tool result falls outside the kept tail.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p heartbit-core --lib agent::pruner::tests::pruned_marker_includes_the_tool_use_id`
Expected: FAIL — the marker (`[pruned: N bytes omitted]`) does not contain `tc_abc`.

- [ ] **Step 3: Thread the ref into the marker** — change the `ToolResult` arm to pass the id to truncation, and update the marker. In `prune_old_tool_results`, replace the `truncate_with_marker(content, max)` call with `truncate_with_marker(content, max, tool_use_id)`, and change `truncate_with_marker`'s signature + marker line:

```rust
fn truncate_with_marker(content: &str, max_bytes: usize, tool_use_id: &str) -> String {
    if content.len() <= max_bytes {
        return content.to_string();
    }
    let keep = floor_char_boundary(content, max_bytes);
    let omitted = content.len() - keep;
    format!(
        "{}\n[pruned: {omitted} bytes omitted, id={tool_use_id} — call fetch_full_output(\"{tool_use_id}\") to restore]",
        &content[..keep]
    )
}
```

> Keep the head-slice logic the existing function used (head-only, or head+tail). The only required change is (a) the new `tool_use_id` param and (b) embedding it in the marker. Update the single call site in the `ToolResult` arm to pass `tool_use_id`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p heartbit-core --lib agent::pruner::tests::pruned_marker_includes_the_tool_use_id`
Expected: PASS. Also run the whole pruner module to catch any other marker assertions that need updating: `cargo test -p heartbit-core --lib agent::pruner`

- [ ] **Step 5: Commit**

```bash
cargo fmt --all -- --check && cargo clippy -p heartbit-core --all-targets -- -D warnings
git add crates/heartbit-core/src/agent/pruner.rs
git commit -m "feat(pruner): truncation marker names the ref (fetch_full_output)"
```

---

## Task 9: runner indexing hook + builder wiring + system-prompt hint

**Files:**
- Modify: `crates/heartbit-core/src/agent/runner.rs` (field + indexing hook)
- Modify: `crates/heartbit-core/src/agent/builder.rs` (field + setter + build transfer + hint)
- Test: `crates/heartbit-core/src/agent/mod.rs` test module (integration), mirroring the existing `MockProvider` event tests

- [ ] **Step 1: Add the runner field** — in `struct AgentRunner<P>` (near the other `Option<Arc<...>>` fields, e.g. after `memory`):

```rust
    pub(super) context_recall_store: Option<Arc<crate::agent::context_recall::ContextRecallStore>>,
```

- [ ] **Step 2: Default it in the builder's `AgentRunnerBuilder` struct + new()** — add the field to the builder struct (in `builder.rs`):

```rust
    pub(super) context_recall_store: Option<Arc<crate::agent::context_recall::ContextRecallStore>>,
```

and initialize it `None` wherever the builder's other `None` fields are initialized (the builder constructor in `runner.rs` ~line 273-310 — add `context_recall_store: None,`).

- [ ] **Step 3: Add the builder setter** — in `impl AgentRunnerBuilder` (`builder.rs`, near `enable_reflection`):

```rust
    /// Enable restore-on-demand: share a `ContextRecallStore` so tool outputs are
    /// indexed for `fetch_full_output` / `recall_context`. Pass the SAME store
    /// into `BuiltinToolsConfig.context_recall_store` so the tools are registered.
    pub fn context_recall_store(
        mut self,
        store: Arc<crate::agent::context_recall::ContextRecallStore>,
    ) -> Self {
        self.context_recall_store = Some(store);
        self
    }
```

- [ ] **Step 4: Transfer it in `build()`** — in the `AgentRunner { ... }` construction inside `build()` (`builder.rs`, near `memory: self.memory,`):

```rust
            context_recall_store: self.context_recall_store,
```

- [ ] **Step 5: Append the system-prompt hint in `build()`** — define a const (top of `builder.rs`):

```rust
/// Appended to the system prompt when restore-on-demand is enabled, so the model
/// knows to act on a pruned marker.
const CONTEXT_RECALL_HINT: &str = "\n\nNote: old tool outputs may be truncated with a \
    '[pruned: … id=<ref>]' marker. Call fetch_full_output(<ref>) to restore the exact content, \
    or recall_context(<query>) to find older outputs by meaning.";
```

and, where `system_prompt` is assembled in `build()` (it's a `let mut system_prompt = …` ~line 672), append after it is finalized:

```rust
        if self.context_recall_store.is_some() {
            system_prompt.push_str(CONTEXT_RECALL_HINT);
        }
```

- [ ] **Step 6: Add the indexing hook** — in `runner.rs`, immediately after the `self.audit(AuditRecord { … "tool_result" … }).await;` block in the tool-result loop (~line 2444, where `output.content`, `call_names[idx]`, `call_ids[idx]` are in scope):

```rust
            if let Some(store) = &self.context_recall_store {
                store.index(&call_ids[idx], &call_names[idx], &output.content).await;
            }
```

- [ ] **Step 7: Write the failing integration test** — add to the `#[cfg(test)] mod tests` in `crates/heartbit-core/src/agent/mod.rs`, mirroring the existing `MockProvider` tool-call tests (see `on_event_emits_tool_call_events`). The mock must emit one tool call to a registered tool, then a final text turn:

```rust
    #[tokio::test]
    async fn tool_output_is_indexed_into_the_context_recall_store() {
        use crate::agent::context_recall::ContextRecallStore;
        use crate::tool::builtins::{builtin_tools, BuiltinToolsConfig};

        let store = Arc::new(ContextRecallStore::new());
        // A provider that calls a read-only builtin once (id "tc_x"), then ends.
        let provider = Arc::new(MockProvider::new(vec![
            MockProvider::tool_call_response("tc_x", "glob", serde_json::json!({"pattern": "*.md"})),
            MockProvider::text_response("done", 1, 1),
        ]));
        let tools = builtin_tools(BuiltinToolsConfig {
            context_recall_store: Some(store.clone()),
            ..Default::default()
        });
        let runner = AgentRunner::builder(provider)
            .name("idx")
            .system_prompt("sys")
            .tools(tools)
            .context_recall_store(store.clone())
            .max_turns(3)
            .build()
            .unwrap();

        runner.execute("go").await.unwrap();

        // The tool's full output was indexed under its tool_call_id.
        assert!(
            store.get("tc_x").await.is_some(),
            "the tool output should be indexed by tool_call_id"
        );
    }
```

> Use the project's actual `MockProvider` tool-call constructor (check `agent/test_helpers.rs` / nearby tests for the exact helper name and signature — e.g. `tool_call_response(id, name, input)`; if it differs, build the `CompletionResponse` literal with a `ContentBlock::ToolUse { id: "tc_x", name: "glob", input }` as the existing tool-call tests do). Pick any registered read-only builtin (`glob`/`list`) that returns without approval.

- [ ] **Step 8: Run test to verify it fails, then passes**

Run: `cargo test -p heartbit-core --lib agent::tests::tool_output_is_indexed_into_the_context_recall_store`
Expected: first FAIL (before Steps 1-6 compile/wire), then PASS once the field, setter, transfer, and hook are in place.

- [ ] **Step 9: Commit**

```bash
cargo fmt --all -- --check && cargo clippy --workspace --exclude mini-crm --all-targets -- -D warnings
git add crates/heartbit-core/src/agent/runner.rs crates/heartbit-core/src/agent/builder.rs crates/heartbit-core/src/agent/mod.rs
git commit -m "feat(agent): index tool outputs into ContextRecallStore + restore-hint prompt"
```

---

## Task 10: end-to-end — prune then restore by the marker's ref

**Files:**
- Test: `crates/heartbit-core/src/agent/context_recall.rs` (a self-contained integration test using the pruner + the tool)

- [ ] **Step 1: Write the test** — add to the `mod tests` block in `context_recall.rs`:

```rust
    #[tokio::test]
    async fn pruned_then_restored_by_marker_ref_roundtrips() {
        use crate::agent::pruner::{prune_old_tool_results, SessionPruneConfig};
        use crate::llm::types::{ContentBlock, Message, Role};
        use crate::tool::Tool;
        use std::sync::Arc;

        // 1. A big tool output is produced and indexed under its id.
        let store = Arc::new(ContextRecallStore::new());
        let big = "RESTORE_ME ".repeat(500); // > prune cap
        store.index("tc_abc", "bash", &big).await;

        // 2. The pruner truncates it in a request view; the marker names the ref.
        let messages = vec![
            Message::user("task"),
            Message { role: Role::User, content: vec![ContentBlock::ToolResult {
                tool_use_id: "tc_abc".into(),
                content: big.clone(),
                is_error: false,
            }]},
            Message::user("r1"),
            Message::assistant("r2"),
            Message::user("r3"),
            Message::assistant("r4"),
        ];
        let (pruned, stats) = prune_old_tool_results(&messages, &SessionPruneConfig::default());
        assert!(stats.did_prune());
        let marker: String = pruned.iter().flat_map(|m| m.content.iter())
            .filter_map(|b| match b { ContentBlock::ToolResult { content, .. } => Some(content.clone()), _ => None })
            .collect();
        assert!(marker.contains("tc_abc") && marker.len() < big.len());

        // 3. The agent fetches by the ref → exact full content restored.
        let tool = crate::tool::builtins::test_fetch_tool(store.clone()); // see note
        let out = tool.execute(&crate::ExecutionContext::default(), serde_json::json!({"ref": "tc_abc"})).await.unwrap();
        assert_eq!(out.content, big, "restore returns the exact original content");
    }
```

> `FetchFullOutputTool` is `pub(crate)`; if it isn't reachable from `context_recall.rs`, either (a) make its struct + `store` field `pub(crate)` and import it directly (`use crate::tool::builtins::fetch_full_output::FetchFullOutputTool;` — add `pub(crate) mod fetch_full_output;`), or (b) call `store.get("tc_abc").await` directly to assert the roundtrip without the tool. Prefer (b) if module visibility is awkward — the tool itself is already covered by Task 5.

- [ ] **Step 2: Run test to verify it passes** (it exercises already-built pieces):

Run: `cargo test -p heartbit-core --lib agent::context_recall::tests::pruned_then_restored_by_marker_ref_roundtrips`
Expected: PASS.

- [ ] **Step 3: Full gate + commit**

```bash
cargo fmt --all -- --check && cargo clippy --workspace --exclude mini-crm --all-targets -- -D warnings && cargo test --workspace --exclude mini-crm
git add crates/heartbit-core/src/agent/context_recall.rs
git commit -m "test(agent): end-to-end prune→restore-by-ref roundtrip"
```

---

## Out of scope / follow-ups (not in this plan)

- **TUI/CLI enablement.** This plan wires the core (config field, builder, runner, tools). Turning it on in the TUI means: create one `Arc<ContextRecallStore>`, pass it to both `BuiltinToolsConfig.context_recall_store` and `AgentRunnerBuilder::context_recall_store`, and (to make it useful) enable the session pruner. Do this as a separate change once the core is proven.
- **Leverage #2** (proactive/append-only compaction defaults) — explicitly sequenced *after* this; reversibility is the precondition.
- **Eviction policy note:** the store is bounded via `with_max_entries(256)`, relying on `InMemoryStore`'s existing eviction. If a future test shows eviction drops recent entries, switch the store to recency-protected eviction or raise the cap.

## Spec coverage check

- §4.1 `ContextRecallStore` → Tasks 2,3 · `InMemoryStore::get` → Task 1 · `FetchFullOutputTool` → Task 5 · `RecallContextTool` → Task 6 · indexing hook → Task 9 · ref-marker → Task 8 · wiring/flag-gating → Tasks 7,9 · prompt hint → Task 9.
- §8 test slices 1-7 → Tasks 1,2,3,5,6,8,9,10 respectively.
- §5 "off = zero overhead" → Task 7 (tools absent without store) + Task 9 (indexing & hint gated on `Some`).
