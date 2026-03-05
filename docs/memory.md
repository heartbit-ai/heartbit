# Memory System

MemGPT-inspired memory with composite recall scoring and hybrid retrieval.

## Storage Backends

| Backend | Config `type =` | Description |
|---------|----------------|-------------|
| `InMemoryStore` | `"in_memory"` | In-process, no persistence |
| `PostgresMemoryStore` | `"postgres"` | PostgreSQL with pgvector for vector search |
| `NamespacedMemory` | — | 3-tier wrapper (user/agent/session) for multi-tenant |

## Memory Types

| Type | Description |
|------|-------------|
| `Episodic` | Event-based memories (default) |
| `Semantic` | Factual knowledge, consolidated from episodic |
| `Reflection` | Meta-observations about patterns |

## Confidentiality Levels

`Public`, `Internal`, `Confidential`, `Restricted` — controls visibility in LLM context.

## Recall Scoring

- **BM25** keyword search with 2x boost
- **Park et al. composite** scoring: recency + importance + relevance + strength
- **Hybrid retrieval**: BM25 + vector cosine similarity fused via Reciprocal Rank Fusion (RRF)

## Ebbinghaus Strength Decay

`effective_strength()` with decay rate of 0.005/hr (~6-day half-life). Strength reinforced +0.2 on access, capped at 1.0.

## Reflection

`ReflectionTracker` monitors cumulative importance. When the threshold is exceeded, it triggers a reflection prompt that produces `Reflection`-type memories.

## Consolidation

`ConsolidationPipeline` clusters entries by Jaccard keyword similarity and merges clusters into `Semantic` entries.

## Pruning

- Auto-prune weak memories at session end (configurable min strength + min age)
- Session pruning: `SessionPruneConfig` auto-trims old tool results before LLM calls
- Pre-compaction flush: extracts tool results to episodic memory before context summarization

## Agent Tools

5 agent-facing tools:

| Tool | Description |
|------|-------------|
| `memory_store` | Store a new memory entry |
| `memory_recall` | Search memories by query |
| `memory_update` | Update an existing memory |
| `memory_forget` | Remove a memory entry |
| `memory_consolidate` | Trigger consolidation pipeline |

## Embedding Providers

Embeddings enable hybrid retrieval (BM25 + vector cosine) for improved recall quality.

| Provider | Config `provider =` | Requirements | Dimension |
|----------|-------------------|--------------|-----------|
| `NoopEmbedding` | `"none"` | None | 0 (BM25-only fallback) |
| `OpenAiEmbedding` | `"openai"` | `OPENAI_API_KEY` | 1536 (small) / 3072 (large) |
| `LocalEmbeddingProvider` | `"local"` | `local-embedding` feature | 384 (MiniLM) and others |

**Local embeddings** run entirely offline via [fastembed](https://github.com/Anush008/fastembed-rs) (ONNX Runtime). Models are downloaded once on first use (~30MB).

Supported local models: `all-MiniLM-L6-v2` (default), `all-MiniLM-L12-v2`, `BGE-small-en-v1.5`, `BGE-base-en-v1.5`, `BGE-large-en-v1.5`, `nomic-embed-text-v1`, `nomic-embed-text-v1.5` (plus quantized variants with `-q` suffix).

## Configuration

```toml
[memory]
type = "in_memory"                    # or "postgres"
# database_url = "postgresql://localhost/heartbit"  # postgres only

[memory.embedding]
provider = "local"                    # "openai", "local", or "none"
model = "all-MiniLM-L6-v2"
cache_dir = "/tmp/fastembed"          # local provider only
```
