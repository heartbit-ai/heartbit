# Memory

An agent without memory rebuilds its world from scratch on every run.
Wiring memory into an `AgentRunner` gives it durable knowledge it can
reach across runs — what the user prefers, what worked last time, what
the current project is about. The framework treats memory as a small,
explicit interface so you can plug in whatever backing store fits your
deployment, from a `HashMap` in tests to PostgreSQL in production.

## The Memory trait

[`Memory`](https://docs.rs/heartbit-core/latest/heartbit_core/memory/trait.Memory.html)
is six async methods — `store`, `recall`, `update`, `forget`,
`add_link`, `prune`. Each returns a
`Pin<Box<dyn Future<Output = Result<…, Error>> + Send + '_>>`, the same
shape used elsewhere in the framework, so the trait is dyn-compatible
and an agent can hold it as `Arc<dyn Memory>`.

`store` and `update` mutate, `recall` reads, `forget` deletes, and
`add_link` connects two entries bidirectionally. `prune` removes
entries whose strength has decayed below a threshold and that are
older than a minimum age — strength decay is covered below. Default
implementations of `add_link` and `prune` are no-ops, so simple stores
only need to implement four methods.

## InMemoryStore and NamespacedMemory

[`heartbit_core::memory::InMemoryStore`](https://docs.rs/heartbit-core/latest/heartbit_core/memory/struct.InMemoryStore.html)
is a thread-safe `HashMap`-backed implementation suitable for tests
and single-process deployments. Construct it with `InMemoryStore::new()`,
hand it to your agent as `Arc::new(store)`, and you're done. It
implements the full trait, including BM25 recall ranking and pruning.

[`heartbit_core::memory::NamespacedMemory`](https://docs.rs/heartbit-core/latest/heartbit_core/memory/struct.NamespacedMemory.html)
wraps any `Memory` implementation and prefixes every entry's `agent`
field with a tenant or user namespace, so a single shared backing
store can serve many isolated agents safely. This is how the
multi-tenant deployments in the daemon mode keep one user's memories
out of another user's recall results — and `prune` is namespace-scoped
by the same prefix so cleanup never reaches across tenants.

## Memory entries

A `MemoryEntry` carries `id`, `agent`, `content`, `category`, and
`tags` for the basics; `created_at`, `last_accessed`, and `access_count`
for usage tracking; and a richer set of fields for ranking and
governance: `importance` (1–10, set by the agent at store time),
`memory_type` (`Episodic`, `Semantic`, or `Reflection`), `keywords`,
`summary`, `strength`, `related_ids`, `source_ids`, an optional
vector `embedding`, `confidentiality`, and `author_user_id` /
`author_tenant_id` for multi-tenant authorship.

The `strength` field is the centerpiece of the framework's
forgetting-curve model. It starts at 1.0, decays exponentially over
time (Ebbinghaus, with a default rate of `0.005` per hour — roughly a
six-day half-life), and is reinforced by `+0.2` (capped at 1.0) on
every successful recall. The pruner uses *effective* strength —
current value minus elapsed decay — so unused entries fade and useful
ones stay sharp.

## The 5 memory tools

When you wire a `Memory` into an agent via
`AgentRunnerBuilder::memory(...)`, five tools become available
automatically — implementing the [MemGPT](https://arxiv.org/abs/2310.08560)
pattern of letting the LLM manage its own knowledge:

- `memory_store` — write a new entry with content, category, tags,
  importance, and keywords.
- `memory_recall` — retrieve entries by text query, category, or tags;
  ranked and limited.
- `memory_update` — edit the content of an existing entry by id.
- `memory_forget` — delete an entry by id.
- `memory_consolidate` — cluster recent episodic entries by keyword
  overlap and merge them into a `Semantic` entry, citing source ids.

The agent decides when to call each: it stores facts as it learns
them, recalls before answering, and consolidates when the episodic
log grows too long.

## Recall and ranking

Recall uses BM25 keyword scoring — with a 2× boost for matches against
the entry's `keywords` field — fused with a composite score that
blends recency, `importance`, relevance, and effective `strength`.
The result is that newer, more important, more frequently-recalled
entries rise to the top, while stale low-confidence entries sink even
if their literal match score is high.

## Embeddings and hybrid search

`Memory` is a trait, so vector search is layered on rather than baked
in. The
[`EmbeddingMemory`](https://docs.rs/heartbit-core/latest/heartbit_core/memory/embedding/struct.EmbeddingMemory.html)
wrapper takes any `Memory` plus an `EmbeddingProvider` and computes
embeddings on `store`, populates `query_embedding` on `recall`, and
fuses BM25 with cosine similarity via Reciprocal Rank Fusion.

For embeddings without an external API call, enable the
`local-embedding` feature on the [heartbit](https://crates.io/crates/heartbit)
umbrella crate. It bundles a fastembed (ONNX Runtime) backend with
the `AllMiniLML6V2` model by default, which is small enough to run on
CPU and gives you private, offline semantic search.

## Memory lifecycle

The end-to-end shape is short — store, recall, update, prune:

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/memory.rs}}
```

In a real agent, the LLM drives this loop through the five memory
tools — you just hand the runner an `Arc<dyn Memory>` and it does
the orchestration.

## Postgres-backed memory

For multi-process deployments and durability across restarts, the
[heartbit](https://crates.io/crates/heartbit) umbrella crate exposes a
PostgreSQL-backed memory store behind the `postgres` feature. The
schema auto-migrates on first connect, so you point at an empty
database and the store creates the tables it needs. The
[Production Considerations](../production/README.md) chapter covers
the connection pool, indexing strategy, and the additional knobs
relevant to running a fleet of agents against a shared store.
