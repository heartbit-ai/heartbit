# heartbit-rs:x — framework-evangelism persona

**Status:** awaiting approval (2026-05-09)
**Predecessor:** heartbit-ghost P1.3g (tool-result redaction + figurative image prompt) merged to `main` at `142d13d`
**Branch:** `feat/heartbit-rs-persona` (to be created off `main`)
**Brainstorming:** done inline 2026-05-09 (this conversation, before this doc)

---

## 1. Goal

Add a second persona, `heartbit-rs:x`, that posts on the same X account as `heartbit-ghost:x` but with a different editorial charter: **demonstrate features of `heartbit-core` and `heartbit-cli` by example**, framework-wide, with every claim grounded in a real file path or type from the repo. Pure on-demand: invoked by `heartbit persona run heartbit-rs:x --review --once "<topic>"`.

The persona reuses the existing review pipeline (`run_review_pipeline`) end-to-end. Only two pipeline pieces change for this persona:

1. The `researcher` agent is swapped for a new `repo_researcher` whose primary tool is a new `repo_inspect` builtin (with `websearch` + `webfetch` available as secondary context for comparisons / external citations).
2. The `writer` recipe receives a persona-specific `system_prompt_addendum` that imposes the evangelism shape *hook → demo → payoff* and forbids release-historian framing.

All other recipes (`style_critic`, `fact_check`, `judge`, `image_generator`, `publisher`) and the X tools are reused as-is — they are persona-agnostic.

This is **not** a release-historian. It does not poll commits or CHANGELOG entries on a schedule. The user picks a topic (or asks for "auto: pick a feature from the menu") and the pipeline produces a thread that *demonstrates* that feature.

## 2. Architecture

```
heartbit persona run heartbit-rs:x --review --once "<topic>"
  ↓
expand persona → AgentConfig set:
  researcher_agent = "repo_researcher"          (per-persona override)
  writer.system_prompt = blend + evangelism addendum
  ↓
run_review_pipeline:
  repo_researcher           ← NEW agent, uses repo_inspect (+ websearch/webfetch optional)
    ↓ digest of canonical file paths, key types, code excerpts, payoff
  writer (×3 candidates)    ← evangelism addendum forces hook→demo→payoff shape
    ↓
  style_critic → fact_check → judge ...
    ↓
  Telegram review (existing) → user picks
    ↓
  image_generator (figurative, existing P1.3g prompt)
    ↓
  publisher → twitter_thread (image attached to head, existing)
```

The new tool `repo_inspect` lives in `crates/heartbit-ghost/src/tools/` (not in `heartbit-core`'s builtins, because it is persona-domain-specific — it reads *this* repo). It is wired only when the persona's `tools_for_persona` declares it.

### 2.1 Why not put `repo_inspect` in `heartbit-core` builtins?

Two reasons:

- **Domain coupling.** `repo_inspect` is hard-coded to the heartbit repo layout (only `crates/heartbit-core/` and `crates/heartbit-cli/` are reachable). A `heartbit-core` builtin should be domain-agnostic.
- **Security boundary.** `read_file` and `grep_repo` over a project's source tree is a credible exfil path if exposed to *any* persona. Keeping it in the ghost crate scopes it to personas that opt in.

A future generalized `code_search` builtin in `heartbit-core` is conceivable but out of scope here.

## 3. Files

| Path | Action | Purpose |
|------|--------|---------|
| `crates/heartbit-ghost/src/tools/repo_inspect.rs` | NEW | `RepoInspectTool` with 4 ops; ~8 wiremock-free unit tests using `tempfile`. |
| `crates/heartbit-ghost/src/tools/mod.rs` | MODIFY | Re-export `RepoInspectTool`. |
| `crates/heartbit-ghost/src/agents/repo_researcher.rs` | NEW | `repo_researcher_recipe()` returning `AgentConfig`; ~3 unit tests. |
| `crates/heartbit-ghost/src/agents/mod.rs` | MODIFY | Register `repo_researcher` recipe; resolve `researcher_agent` field at expand. |
| `crates/heartbit-ghost/data/heartbit-rs-features.toml` | NEW | Curated feature menu (~18 entries). |
| `crates/heartbit-ghost/src/persona/recipe.rs` (or wherever `Recipe` lives) | MODIFY | Add `researcher_agent: Option<String>` and `system_prompt_addendum: Option<String>` fields with `#[serde(default)]`. |
| `crates/heartbit-ghost/src/persona/mod.rs` | MODIFY | `expand()` honors the two new fields. |
| `~/.heartbit/ghost/personas/heartbit-rs:x.toml` | NEW (user-side) | Persona instance config. |
| `~/.heartbit/ghost/corpora/burntsushi.jsonl` | NEW (user-side) | Ingested corpus. |
| `~/.heartbit/ghost/corpora/simonw.jsonl` | NEW (user-side) | Ingested corpus. |

## 4. The `repo_inspect` tool

**Location:** `crates/heartbit-ghost/src/tools/repo_inspect.rs`

**Constructor:**
```rust
pub struct RepoInspectTool {
    repo_root: PathBuf,           // canonical, used for path-escape checks
    allowed_prefixes: Vec<PathBuf>, // ["crates/heartbit-core", "crates/heartbit-cli"]
    feature_menu: FeatureMenu,    // loaded from data/heartbit-rs-features.toml at construction
    max_file_lines: usize,        // 1000
    max_grep_hits: usize,         // 100
}

impl RepoInspectTool {
    pub fn new(repo_root: impl Into<PathBuf>) -> Result<Self, Error>;
}
```

**Operations** (selected via `op` field in tool input):

```jsonc
// op: "read_file"
{ "op": "read_file", "path": "crates/heartbit-core/src/tool/mod.rs", "range": [1, 80] }
// → { "path": "...", "lines": "1: use ...\n2: ..." } or error if path escapes / too big

// op: "grep_repo"
{ "op": "grep_repo", "pattern": "pub trait Tool", "glob": "*.rs" }
// → { "hits": [{"file": "...", "line": 42, "preview": "pub trait Tool: ..."}] }
// Implementation: spawn `git grep -n -- :(top)<allowed_prefixes>` capped at 100 hits.
// (`git grep` respects .gitignore; falls back to `grep -rn` if .git absent — never used in tests.)

// op: "list_features"
{ "op": "list_features" }
// → { "features": [{"name": "tool_trait", "description": "...", "payoff": "..."}, ...] }

// op: "feature_demo"
{ "op": "feature_demo", "name": "tool_trait" }
// → { "name": "tool_trait", "description": "...", "canonical_file": "crates/heartbit-core/src/tool/mod.rs",
//     "key_types": ["Tool", "ToolDefinition", "ToolOutput"], "payoff": "..." }
```

**Path safety.** `read_file` resolves `repo_root.join(path).canonicalize()` and rejects if the result does not start with one of `allowed_prefixes` (after canonicalization too). Symlink escapes, `..`, and absolute paths all fail. `grep_repo` runs `git grep` with explicit `:(top)<prefix>` pathspecs — same restriction.

**Size caps.** `read_file` rejects files >1000 lines without `range`; with `range` it accepts any range up to 1000 lines wide. `grep_repo` truncates to 100 hits and notes truncation.

### 4.1 The feature menu schema

`crates/heartbit-ghost/data/heartbit-rs-features.toml`:

```toml
version = 1

[[feature]]
name = "tool_trait"
description = "The Tool trait — definition() + execute() — that powers everything in heartbit-core"
canonical_file = "crates/heartbit-core/src/tool/mod.rs"
key_types = ["Tool", "ToolDefinition", "ToolOutput"]
payoff = "implement two methods, get a fully-wired tool with retry, guardrails, telemetry"

[[feature]]
name = "agent_runner"
description = "Standalone agent loop with tokio::JoinSet for parallel tool execution"
canonical_file = "crates/heartbit-core/src/agent/runner.rs"
key_types = ["AgentRunner", "AgentRunnerBuilder", "AgentOutput"]
payoff = "single-process agent loop, no Restate / no daemon — drop into any tokio app"

# … 16 more entries — see Appendix A for the full V1 list
```

The menu is loaded once at `RepoInspectTool::new()` and held in memory. Reloading requires restarting the process — acceptable for a curated, manually-maintained file.

**Initial V1 menu (18 features)** — listed in Appendix A.

## 5. The `repo_researcher` agent

**Location:** `crates/heartbit-ghost/src/agents/repo_researcher.rs`

**Recipe shape:**
- `name`: `"repo_researcher"`
- `description`: "Find substance about a heartbit-rs feature: canonical file, code excerpt, payoff."
- `max_turns`: 25 (slightly higher than `researcher` to allow exploration of grep + read iteration)
- `max_tokens`: 4096
- `reasoning_effort`: `"medium"`
- `response_schema`: None (free-form digest)
- `tools`: `repo_inspect`, `websearch`, `webfetch`

**System prompt** (full text):
```
You are a research analyst for a Rust agent framework called heartbit-rs.
Given a feature name or topic, find the substance: the canonical file
where it lives, the key types, a representative code excerpt, and a
one-sentence payoff for someone reading about it.

PROCESS
1. If the user named a feature in the menu (e.g., "tool_trait", "memory_bm25"),
   call `repo_inspect` with `op: "feature_demo"` and read the canonical_file.
2. If the user gave a free-form topic, call `repo_inspect` with `op: "list_features"`
   first to see what's available, then either pick the closest one or use
   `op: "grep_repo"` to locate definitions yourself.
3. Read at most 2-3 files; pick the smallest excerpt that demonstrates the
   feature (typically a trait definition, a struct + 1-2 methods, or a single
   public function). Aim for ≤30 lines per excerpt.
4. `websearch` / `webfetch` are available ONLY for OPTIONAL external context
   (e.g. "how this compares to LangGraph", "the original paper"). They are
   never the primary substance. The substance always comes from the repo.

OUTPUT FORMAT (free-form text, no JSON):
- Feature name + 1-sentence framing.
- Canonical file path (e.g., `crates/heartbit-core/src/tool/mod.rs`).
- Key types: comma-separated list.
- Code excerpt: ≤30 lines, fenced ```rust block, with the line numbers if from
  a range.
- Payoff: 1-2 sentences on what this enables for someone using the framework.
- Optional: 1-2 external context bullets with sources.

Do NOT write the post. The writer composes. Do NOT speculate beyond what
the files show.
```

## 6. Writer prompt addendum (persona-level)

The persona's TOML adds a `[recipe.overrides] system_prompt_addendum` (multiline TOML string) that the writer's expand step appends to the blended prompt:

```
EVANGELISM MODE — heartbit-rs:x

You are showing what heartbit-rs (a Rust multi-agent framework) does, by
example. Your audience is Rust developers and AI engineers evaluating
the framework.

THREAD SHAPE
Every thread is structured as: hook → demo → payoff.
- Hook: ONE concrete sentence stating what this feature lets you do
  (e.g. "Implement two methods on a trait, get a fully-wired tool with
  retry, guardrails, and telemetry.").
- Demo: a code excerpt taken from the researcher's digest. Paraphrase
  for tweet-friendliness if needed but do not invent code that wasn't
  in the digest. Reference the canonical file path inline (e.g.,
  "in `crates/heartbit-core/src/tool/mod.rs`") so curious readers can
  cross-check.
- Payoff: 1-2 tweets on what this enables — concrete benefits,
  not adjectives.

GROUND TRUTH
- Every claim about heartbit-rs MUST trace back to a real file path or
  type the researcher surfaced. No vague "powerful" / "elegant" /
  "production-grade" framework adjectives without the corresponding
  code.
- If you cannot ground a claim, drop the claim.

NEVER
- Release-note framing ("we shipped X yesterday", "new in v2.0", "just
  released"). Frame everything time-agnostically — "here's what X does"
  not "here's what we just added".
- Marketing superlatives without code backing them.
- Code excerpts longer than 8 lines per tweet (it gets truncated by X
  and unreadable on mobile).
```

## 7. Config plumbing

Two new fields on the persona's `Recipe` (or whatever the existing serde struct is called):

```rust
#[derive(Deserialize, Serialize)]
pub struct RecipeConfig {
    // ... existing fields ...
    #[serde(default)]
    pub researcher_agent: Option<String>,  // None → "researcher" (default); Some("repo_researcher") for heartbit-rs:x
    // ... existing fields ...
}

#[derive(Deserialize, Serialize)]
pub struct RecipeOverrides {
    // ... existing fields ...
    #[serde(default)]
    pub system_prompt_addendum: Option<String>,
}
```

### 7.1 `expand()` changes

Two changes to the persona's `expand()` (which builds the `AgentConfig` set):

1. The researcher slot resolves via `match recipe.researcher_agent.as_deref() { Some("repo_researcher") => repo_researcher_recipe(), _ => researcher_recipe() }`. Unknown values produce a clear error at expand-time, not at runtime. (Eventually this could become a registry; for now a single match arm is fine.)

2. The writer's system prompt is `format!("{}\n\n{}", blended_prompt, addendum)` when `addendum` is `Some`.

Both fields are `#[serde(default)]` so existing personas (heartbit-ghost:x) still parse and behave unchanged.

## 8. Persona instance TOML

`~/.heartbit/ghost/personas/heartbit-rs:x.toml`:

```toml
version = 1

[recipe]
version = 1
researcher_agent = "repo_researcher"

[[recipe.blend]]
writer = "burntsushi"
weight = 0.5

[[recipe.blend]]
writer = "simonw"
weight = 0.5

[recipe.overrides]
thread_max_length = 12
ai_tells_to_avoid = [
    "delve", "leverage", "unlock", "cutting-edge", "revolutionary",
    "game-changing", "—", "–"
]

[recipe.overrides.formatting]
em_dashes = "forbidden"
periods = "always"
quotation_marks = "double"
line_breaks = "single"

[recipe.overrides.system_prompt_addendum]
# multiline string — see §6 for full text
text = """
EVANGELISM MODE — heartbit-rs:x
...
"""
```

(Exact TOML shape for `system_prompt_addendum` may need adjustment — TOML multiline-string-as-leaf vs `text = """..."""` table-of-one-key. Plan task verifies.)

## 9. Corpus ingestion

Two corpora to ingest manually before live tests:

```
~/.heartbit/ghost/corpora/burntsushi.jsonl    (Andrew Gallant — ripgrep / regex / xsv)
~/.heartbit/ghost/corpora/simonw.jsonl        (Simon Willison — datasette / llm CLI)
```

Use the same workflow that fed `karpathy.jsonl` and `bcherny.jsonl`. Aim for ~300-500 tweets per author covering technical / shipping / framework-author content (filter out personal, political, retweet-only). Then:

```bash
heartbit persona corpus add heartbit-rs:x burntsushi ~/.heartbit/ghost/corpora/burntsushi.jsonl
heartbit persona corpus add heartbit-rs:x simonw    ~/.heartbit/ghost/corpora/simonw.jsonl
heartbit persona profile rebuild heartbit-rs:x
```

(Corpus ingestion is user-side, not part of the implementation plan; it is a precondition for live tests.)

## 10. Tests

| Layer | Count | What it covers |
|-------|-------|----------------|
| `repo_inspect` unit | 8 | path canonicalization, allowed_prefix enforcement, range bounds, file-too-big, glob filter, list_features deserializes from real fixture, feature_demo lookup, missing-feature error |
| `repo_researcher` unit | 3 | recipe shape (name/max_turns/max_tokens), system prompt mentions repo_inspect, system prompt explicitly forbids websearch as primary substance |
| `Recipe` config | 2 | `researcher_agent` parses with `#[serde(default)]`, `system_prompt_addendum` is appended in `expand()` |
| Integration | 1 | `expand("heartbit-rs:x")` returns AgentConfig set with `repo_researcher` in researcher slot and addendum applied to writer prompt |
| Live (manual) | 1 | end-to-end `heartbit persona run heartbit-rs:x --review --once "show what the Tool trait gives you"` produces a thread referencing `crates/heartbit-core/src/tool/mod.rs` and at least one of `Tool`, `ToolDefinition`, `ToolOutput` |

Total: ~14 new automated tests + 1 manual live test.

## 11. Out of scope (explicitly deferred)

- **Auto-trigger on commits or cron.** Belongs to a future phase if (and only if) on-demand usage proves insufficient. Adding it now would push back toward the release-historian framing this design rejected.
- **Per-persona X credentials.** Same X account as `heartbit-ghost:x` for V1.
- **`features.toml` auto-generation from rustdoc / `#[promo]` macros.** Manual maintenance is fine for ~18-30 features; revisit only when manual maintenance friction becomes a real blocker.
- **Generalized `code_search` builtin in `heartbit-core`.** Possible future work; would require its own scope/security design.
- **Audit log integration.** Same as `heartbit-ghost:x` — both personas use the existing audit infrastructure unchanged.

## 12. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| `repo_inspect` becomes a code-exfiltration vector if a malicious prompt smuggles file paths to a tool that returns content. | Allowed_prefixes restricts to `crates/heartbit-core/` and `crates/heartbit-cli/`; both are open-source code. Path canonicalization rejects symlink escapes. No `.env`, no test fixtures with secrets are reachable. |
| The agent loops on `grep_repo` chasing irrelevant matches. | `max_grep_hits=100`, `max_turns=25` on the researcher, plus the system prompt explicitly says "read at most 2-3 files". |
| The feature menu rots as the framework evolves. | Owner adds new entries when shipping a notable feature (~2min per entry). A staleness check (CI test that all `canonical_file` paths exist) catches deletions. Add this CI check as part of the plan. |
| The writer hallucinates code that wasn't in the researcher's digest. | The addendum says "do not invent code that wasn't in the digest". The fact_check agent (already in pipeline) provides a second-pass verification. The Telegram review is the human gate. |
| Tweet quality is uneven across features (some don't make good demos). | The menu acts as a curation layer — only features that *do* demo well land in it. If a feature is hard to demonstrate in a tweet, leave it out of the menu. |

---

## Appendix A — V1 feature menu (18 entries)

Drawn from `MEMORY.md` and `CHANGELOG.md`. Each entry needs: `name` (snake_case identifier), `description` (one sentence), `canonical_file` (single primary file), `key_types` (1-3 types), `payoff` (one sentence on what it enables).

1. `tool_trait` — Tool trait + execute() + ToolOutput. `crates/heartbit-core/src/tool/mod.rs`.
2. `agent_runner` — Standalone agent loop with `tokio::JoinSet`. `crates/heartbit-core/src/agent/runner.rs`.
3. `orchestrator` — Multi-agent dispatch via `DelegateTaskTool` / `FormSquadTool`. `crates/heartbit-core/src/agent/orchestrator.rs` (verify path).
4. `memory_trait` — `Memory` trait with 6 methods (store/recall/update/forget/add_link/prune). `crates/heartbit-core/src/memory/mod.rs`.
5. `memory_bm25` — Optional BM25 inverted index for memory recall via `MemoryQuery::exact_words`. `crates/heartbit-core/src/memory/bm25.rs` (verify path).
6. `guardrails` — `Guardrail` trait with 4 hooks (pre_llm/post_llm/pre_tool/post_tool). `crates/heartbit-core/src/agent/guardrails/mod.rs`.
7. `llm_judge` — `LlmJudgeGuardrail` using a cheap model to vet outputs. `crates/heartbit-core/src/agent/guardrails/llm_judge.rs`.
8. `workflow_agents` — Sequential / Parallel / Loop agents for deterministic orchestration. `crates/heartbit-core/src/agent/workflow.rs`.
9. `mcp_client` — MCP Streamable HTTP client. `crates/heartbit-core/src/mcp/` (verify primary file).
10. `cascading_provider` — Try cheapest model first, escalate on rejection. `crates/heartbit-core/src/llm/cascade.rs`.
11. `retrying_provider` — Wraps any LlmProvider with retry logic. `crates/heartbit-core/src/llm/retry.rs` (verify path).
12. `tool_redaction` — `Tool::redact_for_history` strips large blobs from conversation history. `crates/heartbit-core/src/tool/mod.rs`.
13. `daemon_mode` — Kafka consumer + Axum HTTP + SSE for server deployments. `crates/heartbit/src/daemon/mod.rs` (verify path — heartbit umbrella crate).
14. `prompt_caching` — 3 cache breakpoints (system prompt + last tool def + 2nd-to-last user msg). `crates/heartbit-core/src/llm/anthropic.rs` (verify path).
15. `tool_profiles` — `ToolProfile::Conversational/Standard/Full` to reduce input tokens. `crates/heartbit-core/src/agent/tool_filter.rs`.
16. `doom_loop_detection` — Hash tool-call batches per turn, abort on repetition. `crates/heartbit-core/src/agent/runner.rs` (search `DoomLoopTracker`).
17. `auto_compaction` — On `ContextOverflow`, summarize + retry. `crates/heartbit-core/src/agent/runner.rs` (search `inject_summary`).
18. `restate_workflows` — Durable agent execution with replay. `crates/heartbit-core/src/workflow/agent_workflow.rs` (verify path).

(File paths marked "verify path" are best-guess from MEMORY.md; the implementation plan will confirm or correct each before populating `features.toml`.)

---

**Self-review notes (to fix during plan-writing if any):**
- Spec covers the 11 design decisions made in brainstorming: name (D), corpus (D), scope (B), researcher backend (B+websearch), repo_inspect surface (B), framing (demonstrate-by-example), cadence (A), X account (A), persona name (D), reuse model (share most + new researcher + writer addendum), out-of-scope items.
- One ambiguity: TOML shape for `system_prompt_addendum` — leaf string vs table-with-`text` field. Plan task #1 verifies and picks one.
- One TBD-equivalent: Appendix A's `canonical_file` paths for ~6 features marked "verify path". This is intentional — the implementation plan task that builds `features.toml` is the right place to do that grep, not this design doc.
