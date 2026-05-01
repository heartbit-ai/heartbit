# heartbit-core User Documentation (mdBook) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a comprehensive long-form user guide ("the heartbit book") for `heartbit-core`, hosted at `docs.heartbit.ai` via GitHub Pages. 12 chapters, examples sourced from real `crates/heartbit-core/examples/*.rs` files via mdBook `{{#include}}`, doctests on key public-API entry points, CI auto-deploys on push to `main`.

**Architecture:** mdBook source in `book/` at the repo root. Companion to rustdoc on docs.rs (book = narrative, rustdoc = API reference; they cross-link). 3–4 new example files added in `crates/heartbit-core/examples/` for chapters that lack one; 6–8 doctests added to key entry points. CI: GitHub Actions → mdBook build → `gh-pages` branch → custom domain `docs.heartbit.ai`.

**Tech Stack:** mdBook (latest), `mdbook-mermaid`, `mdbook-linkcheck`, `peaceiris/actions-mdbook@v2`, `peaceiris/actions-gh-pages@v4`. No new Rust deps.

**Spec:** `docs/superpowers/specs/2026-05-01-heartbit-core-user-docs-design.md`

---

## File Map

**Create:**
- `book/book.toml` — mdBook config
- `book/src/SUMMARY.md` — chapter index
- `book/src/introduction.md` — Chapter 1
- `book/src/getting-started/README.md` + 4 sub-pages — Chapter 2
- `book/src/agents/README.md` (+ optional sub-pages) — Chapter 3
- `book/src/tools/README.md` — Chapter 4
- `book/src/memory/README.md` — Chapter 5
- `book/src/guardrails/README.md` — Chapter 6
- `book/src/workflow-agents/README.md` — Chapter 7
- `book/src/orchestration/README.md` — Chapter 8
- `book/src/configuration/README.md` — Chapter 9
- `book/src/eval/README.md` — Chapter 10
- `book/src/recipes/README.md` (+ 1 sub-page per recipe = 6 files) — Chapter 11
- `book/src/production/README.md` — Chapter 12
- `.github/workflows/book.yml` — CI workflow
- `crates/heartbit-core/examples/sequential_agent.rs` — new example for Chapter 7
- `crates/heartbit-core/examples/dag_agent.rs` — new example for Chapter 7
- `crates/heartbit-core/examples/from_toml.rs` — new example for Chapter 9

**Modify:**
- `crates/heartbit-core/src/agent/builder.rs` (or wherever `AgentRunner::builder` is defined) — add doctest
- `crates/heartbit-core/src/tool/mod.rs` — add doctest on `Tool` trait
- `crates/heartbit-core/src/memory/mod.rs` — add doctest on `Memory` trait
- `crates/heartbit-core/src/agent/guardrail.rs` — add doctest on `Guardrail` trait
- `crates/heartbit-core/src/eval/mod.rs` — add doctest on `EvalRunner::new`
- `crates/heartbit-core/src/agent/workflow.rs` — add doctests on `SequentialAgent::new`, `DagAgent::new` (or wherever DagAgent lives), `VotingAgent::new`
- `CONTRIBUTING.md` — one-line note about updating chapters when public APIs change
- `README.md` (top-level) — add a `[![docs](https://img.shields.io/badge/docs-heartbit.ai-blue)](https://docs.heartbit.ai)` badge near the existing crates.io / docs.rs badges

**External (manual, by project owner — not code work):**
- DNS registrar for `heartbit.ai`: add `docs.heartbit.ai. CNAME heartbit-ai.github.io.`
- GitHub repo Settings → Pages → enable, source branch `gh-pages`, custom domain `docs.heartbit.ai`, enforce HTTPS.

---

## Chapter Writing Convention (read once, applies to every chapter task)

Each chapter task asks the implementer to **write prose**, not just paste code. The plan can't embed the full ~600–1,000-word prose for each chapter without exploding to 20k lines, so chapter tasks specify:

1. **The exact file path(s).**
2. **Required section headings** (the chapter's structure — what subtopics it must cover, in order).
3. **Required code blocks** with exact `{{#include}}` paths.
4. **Required cross-links** with exact docs.rs URLs.
5. **Length target** (a soft cap, e.g., "~800 words").
6. **Verification**: `cd book && mdbook build` passes, linkcheck reports zero dead links.

Implementers writing chapter prose follow this voice guide:

- **Active voice, second person** — "You build an agent by..." not "Agents are built by..."
- **Tutorial-grade for early chapters (1–4)**, reference-style with cross-links for later chapters (7–12). Standard "Rust Programming Language" pacing.
- **No emoji, no marketing language.** Match the tone of the rewritten top-level `README.md`.
- **Each new concept gets one well-explained example**; subsequent uses cross-link instead of re-explaining.
- **No fictional code blocks.** Every code block either uses `{{#include}}` to pull from a real example file or is a doctest cross-referenced via link.

Before writing a chapter, the implementer skims the previous chapter's prose to maintain tone continuity.

---

## Task 0: Worktree setup

**Files:** none modified yet.

- [ ] **Step 0.1: Confirm B3 has merged to main.**

```bash
cd /home/pleclech/projects/heartbit
git fetch origin
git log --oneline origin/main -3
```
Expected: top of origin/main shows the B3 squash-merge plus the user-docs spec commit (`02f30b2 docs: heartbit-core user documentation (mdBook) design spec`). If B3 hasn't merged yet, that's fine — this round is independent of B3 and can proceed; the book references heartbit-core APIs that are stable on main regardless.

- [ ] **Step 0.2: Sync local main and create the user-docs branch + worktree.**

```bash
git checkout main
git pull --rebase
git worktree add .worktrees/user-docs -b feat/user-docs
cd .worktrees/user-docs
```

The branch `feat/user-docs` is created off the latest main. Subsequent commits stay local on this branch.

- [ ] **Step 0.3: Verify baseline gate.**

```bash
cargo fmt -- --check && \
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3 && \
cargo test --workspace --no-run 2>&1 | tail -3 && \
cargo test --workspace --lib 2>&1 | grep "test result"
```

Expected: all four exit 0. If any fails, **stop** — main is not green; this round cannot start until that is fixed.

- [ ] **Step 0.4: Install mdBook tooling locally for live preview.**

```bash
cargo install mdbook mdbook-mermaid mdbook-linkcheck
mdbook --version
```

Expected: prints `mdbook v0.4.x` or higher.

---

## Task 1: Scaffold `book/`

**Files:**
- Create: `book/book.toml`
- Create: `book/src/SUMMARY.md`
- Create: `book/src/introduction.md` (placeholder; real content in Task 5)
- Create: `book/.gitignore` (excludes `book/book/` build output)

- [ ] **Step 1.1: Create `book/book.toml`.**

```toml
[book]
title = "The Heartbit Book"
description = "User guide for heartbit-core, the Rust agentic framework"
authors = ["Pascal Le Clech"]
language = "en"
src = "src"

[output.html]
default-theme = "light"
preferred-dark-theme = "navy"
git-repository-url = "https://github.com/heartbit-ai/heartbit"
git-repository-icon = "fa-github"
edit-url-template = "https://github.com/heartbit-ai/heartbit/edit/main/book/{path}"
site-url = "/"

[output.html.search]
enable = true
limit-results = 30
use-boolean-and = true

[preprocessor.mermaid]
command = "mdbook-mermaid"

[preprocessor.linkcheck]
follow-web-links = false
warning-policy = "error"
exclude = [
    # Allow docs.rs URLs even before the crate is published; remove this exclusion
    # once heartbit-core is on crates.io.
    "docs\\.rs/heartbit-core",
]
```

- [ ] **Step 1.2: Create `book/src/SUMMARY.md`.**

```markdown
# Summary

[Introduction](./introduction.md)

# User Guide

- [Getting Started](./getting-started/README.md)
  - [Installation](./getting-started/installation.md)
  - [Hello agent](./getting-started/hello-agent.md)
  - [Choosing an LLM provider](./getting-started/providers.md)
  - [API keys and environment](./getting-started/env.md)
- [Agents](./agents/README.md)
- [Tools](./tools/README.md)
- [Memory](./memory/README.md)
- [Guardrails](./guardrails/README.md)
- [Workflow Agents](./workflow-agents/README.md)
- [Multi-Agent Orchestration](./orchestration/README.md)
- [Configuration](./configuration/README.md)
- [Eval Framework](./eval/README.md)

# Cookbook

- [Recipes](./recipes/README.md)
  - [Chat agent with web search](./recipes/chat-with-search.md)
  - [Code-aware agent](./recipes/code-aware.md)
  - [Multi-agent research workflow](./recipes/multi-agent-research.md)
  - [Long-running agent with persistent memory](./recipes/persistent-memory.md)
  - [Eval-driven prompt iteration](./recipes/eval-driven.md)
  - [MCP server integration](./recipes/mcp-integration.md)

# Operations

- [Production Considerations](./production/README.md)
```

- [ ] **Step 1.3: Create placeholder `book/src/introduction.md`.**

```markdown
# Introduction

> *Placeholder. Real content lands in Task 5.*

This page exists so `mdbook build` succeeds while subsequent tasks add chapter content.
```

- [ ] **Step 1.4: Create placeholder files for every chapter referenced in `SUMMARY.md`.**

mdBook fails the build if `SUMMARY.md` references files that don't exist. Create stubs for each:

```bash
mkdir -p book/src/getting-started book/src/agents book/src/tools book/src/memory \
         book/src/guardrails book/src/workflow-agents book/src/orchestration \
         book/src/configuration book/src/eval book/src/recipes book/src/production

for f in \
    book/src/getting-started/README.md \
    book/src/getting-started/installation.md \
    book/src/getting-started/hello-agent.md \
    book/src/getting-started/providers.md \
    book/src/getting-started/env.md \
    book/src/agents/README.md \
    book/src/tools/README.md \
    book/src/memory/README.md \
    book/src/guardrails/README.md \
    book/src/workflow-agents/README.md \
    book/src/orchestration/README.md \
    book/src/configuration/README.md \
    book/src/eval/README.md \
    book/src/recipes/README.md \
    book/src/recipes/chat-with-search.md \
    book/src/recipes/code-aware.md \
    book/src/recipes/multi-agent-research.md \
    book/src/recipes/persistent-memory.md \
    book/src/recipes/eval-driven.md \
    book/src/recipes/mcp-integration.md \
    book/src/production/README.md \
  ; do
    name=$(basename "$f" .md | sed 's/-/ /g')
    case "$name" in README) name=$(basename $(dirname "$f"));; esac
    printf '# %s\n\n> *Placeholder. Real content lands in subsequent tasks.*\n' "$(echo "$name" | sed 's/.*/\u&/')" > "$f"
done
```

- [ ] **Step 1.5: Create `book/.gitignore`.**

```gitignore
book/
```

(`book/book/` is the mdBook HTML output; not committed.)

- [ ] **Step 1.6: Verify `mdbook build` works.**

```bash
cd book && mdbook build && cd ..
ls book/book/index.html
```
Expected: file exists.

- [ ] **Step 1.7: Verify live preview works.**

```bash
cd book && mdbook serve --port 3000 &
sleep 2 && curl -s http://localhost:3000/ | grep "Heartbit Book"
kill %1
cd ..
```
Expected: prints HTML title containing "Heartbit Book".

- [ ] **Step 1.8: Commit.**

```bash
git add book/ Cargo.toml
git commit -m "feat(book): scaffold mdBook with chapter placeholders

book.toml, SUMMARY.md, and placeholder chapter files for every section
referenced in the table of contents. mdbook build succeeds; subsequent
tasks fill in content."
```

---

## Task 2: Add CI workflow for the book

**Files:**
- Create: `.github/workflows/book.yml`

- [ ] **Step 2.1: Create `.github/workflows/book.yml`.**

```yaml
name: Book

on:
  push:
    branches: [main]
    paths: ['book/**', 'crates/heartbit-core/examples/**', '.github/workflows/book.yml']
  pull_request:
    paths: ['book/**', 'crates/heartbit-core/examples/**', '.github/workflows/book.yml']

permissions:
  contents: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install mdBook
        uses: peaceiris/actions-mdbook@v2
        with:
          mdbook-version: 'latest'

      - name: Cache cargo install
        uses: actions/cache@v4
        with:
          path: ~/.cargo/bin
          key: cargo-mdbook-plugins-v1

      - name: Install mdBook plugins
        run: |
          if ! command -v mdbook-mermaid >/dev/null; then cargo install mdbook-mermaid; fi
          if ! command -v mdbook-linkcheck >/dev/null; then cargo install mdbook-linkcheck; fi

      - name: Build
        run: cd book && mdbook build

      - name: Deploy to gh-pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./book/book
          cname: docs.heartbit.ai
```

- [ ] **Step 2.2: Validate the YAML.**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/book.yml')); print('OK')"
```
Expected: `OK`.

- [ ] **Step 2.3: Commit.**

```bash
git add .github/workflows/book.yml
git commit -m "ci(book): add mdBook build + GitHub Pages deploy

Workflow runs on push to main (deploys to gh-pages with CNAME
docs.heartbit.ai) and on pull_request (build-only, no deploy). Caches
the cargo install of mdBook plugins for fast subsequent runs."
```

---

## Task 3: Add 3 new examples for chapters that lack one

**Files:**
- Create: `crates/heartbit-core/examples/sequential_agent.rs`
- Create: `crates/heartbit-core/examples/dag_agent.rs`
- Create: `crates/heartbit-core/examples/from_toml.rs`

- [ ] **Step 3.1: Create `crates/heartbit-core/examples/sequential_agent.rs`.**

```rust
//! Sequential workflow: chain a researcher agent into a writer agent.
//!
//! `cargo run -p heartbit-core --example sequential_agent`

use std::sync::Arc;

use heartbit_core::{
    AgentRunner, AnthropicProvider, BoxedProvider, RetryingProvider, SequentialAgent,
};

#[tokio::main]
async fn main() -> Result<(), heartbit_core::Error> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let provider = Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(
        AnthropicProvider::new(&api_key, "claude-sonnet-4-20250514"),
    )));

    let researcher = AgentRunner::builder(provider.clone())
        .system_prompt("Research the topic and produce a concise factual summary (3 bullet points).")
        .build()?;

    let writer = AgentRunner::builder(provider)
        .system_prompt("Rewrite the input as a single engaging paragraph for a general audience.")
        .build()?;

    let mut workflow = SequentialAgent::builder()
        .agent(Box::new(researcher))
        .agent(Box::new(writer))
        .build()?;

    let output = workflow.execute("The history of the Rust programming language").await?;
    println!("{}", output.result);
    println!(
        "Total tokens used: {} in / {} out",
        output.tokens_used.input_tokens, output.tokens_used.output_tokens,
    );
    Ok(())
}
```

- [ ] **Step 3.2: Verify it compiles.**

```bash
cargo check --example sequential_agent -p heartbit-core 2>&1 | tail -3
```

If compile errors surface (e.g., the actual `SequentialAgent` API is named differently), inspect `crates/heartbit-core/src/agent/workflow.rs` and adjust the example to match the real API. Common adjustments:
- `SequentialAgent::builder().agent(...)` may be `SequentialAgent::builder().agents(vec![...])`.
- `output.result` may be named `output.content` or `output.text` — read `crates/heartbit-core/src/agent/mod.rs`'s `AgentOutput` struct definition.

Adjust the example until it compiles cleanly. The book chapter (Task 11) will `{{#include}}` whatever the final compiled version looks like.

- [ ] **Step 3.3: Create `crates/heartbit-core/examples/dag_agent.rs`.**

```rust
//! DAG workflow: a planner feeds two parallel workers (research + critique),
//! then a synthesizer combines their outputs.
//!
//! `cargo run -p heartbit-core --example dag_agent`

use std::sync::Arc;

use heartbit_core::{
    AgentRunner, AnthropicProvider, BoxedProvider, DagAgent, RetryingProvider,
};

#[tokio::main]
async fn main() -> Result<(), heartbit_core::Error> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let provider = Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(
        AnthropicProvider::new(&api_key, "claude-sonnet-4-20250514"),
    )));

    let make = |prompt: &str| {
        AgentRunner::builder(provider.clone())
            .system_prompt(prompt)
            .build()
            .unwrap()
    };

    let mut dag = DagAgent::builder()
        .node("planner", Box::new(make("Outline the question into 2 parallel research questions.")))
        .node("research", Box::new(make("Answer the research question concisely.")))
        .node("critique", Box::new(make("Identify weaknesses in the proposed answer.")))
        .node("synthesizer", Box::new(make("Combine research and critique into a final answer.")))
        .edge("planner", "research")
        .edge("planner", "critique")
        .edge("research", "synthesizer")
        .edge("critique", "synthesizer")
        .build()?;

    let output = dag.execute("Should we use Rust for our agent runtime?").await?;
    println!("{}", output.result);
    Ok(())
}
```

- [ ] **Step 3.4: Verify it compiles.**

```bash
cargo check --example dag_agent -p heartbit-core 2>&1 | tail -5
```

If `DagAgent::builder().node(name, agent).edge(from, to).build()` doesn't match the real API, inspect `crates/heartbit-core/src/agent/dag.rs` and adjust. The DAG might use string IDs, integer IDs, or a different builder shape; adapt the example to match.

- [ ] **Step 3.5: Create `crates/heartbit-core/examples/from_toml.rs`.**

```rust
//! Loading agent configuration from a TOML file.
//!
//! `cargo run -p heartbit-core --example from_toml`

use heartbit_core::HeartbitConfig;

fn main() -> Result<(), heartbit_core::Error> {
    let toml_text = r#"
[provider]
provider_type = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "assistant"
system_prompt = "You are a helpful assistant."
max_turns = 10
max_tokens = 4096
"#;

    let config: HeartbitConfig = toml::from_str(toml_text)
        .map_err(|e| heartbit_core::Error::Config(format!("toml parse: {e}")))?;
    config.validate()?;

    println!("Loaded {} agent(s)", config.agents.len());
    for agent in &config.agents {
        println!("  - {} (max_turns={})", agent.name, agent.max_turns);
    }
    Ok(())
}
```

- [ ] **Step 3.6: Verify it compiles.**

```bash
cargo check --example from_toml -p heartbit-core 2>&1 | tail -3
```

If `HeartbitConfig::validate()` doesn't exist or has a different signature, inspect `crates/heartbit-core/src/config/mod.rs` and adjust. Common adjustments: `validate(&self)` may return `Result<()>` directly or be named `check_invariants()`.

- [ ] **Step 3.7: Run full gate to confirm no regressions.**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo check --examples -p heartbit-core 2>&1 | tail -3
```

All three exit 0.

- [ ] **Step 3.8: Commit.**

```bash
git add crates/heartbit-core/examples/sequential_agent.rs \
        crates/heartbit-core/examples/dag_agent.rs \
        crates/heartbit-core/examples/from_toml.rs
git commit -m "feat(examples): add sequential, dag, and from_toml examples

Three new runnable examples for upcoming book chapters that lack one:
- sequential_agent.rs (Workflow Agents chapter)
- dag_agent.rs (Workflow Agents chapter)
- from_toml.rs (Configuration chapter)

Each is ~30-50 LOC, demonstrates one specific API, and compiles via the
existing cargo check --examples gate."
```

---

## Task 4: Add doctests to key public-API entry points

**Files:**
- Modify: `crates/heartbit-core/src/agent/mod.rs` (or wherever `AgentRunner::builder` is defined — search via `grep -rn "impl<P> AgentRunner<P>" crates/heartbit-core/src/`)
- Modify: `crates/heartbit-core/src/tool/mod.rs` (`Tool` trait definition)
- Modify: `crates/heartbit-core/src/memory/mod.rs` (`Memory` trait definition)
- Modify: `crates/heartbit-core/src/agent/guardrail.rs` (`Guardrail` trait definition)
- Modify: `crates/heartbit-core/src/eval/mod.rs` (`EvalRunner::new`)
- Modify: `crates/heartbit-core/src/agent/workflow.rs` (`SequentialAgent::builder`, `VotingAgent::new`)

For each modification, add a `///` doctest above the relevant item. The pattern:

```rust
/// Build an agent.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use heartbit_core::{AgentRunner, AnthropicProvider, BoxedProvider};
///
/// # async fn run() -> Result<(), heartbit_core::Error> {
/// let provider = Arc::new(BoxedProvider::new(
///     AnthropicProvider::new("sk-...", "claude-sonnet-4-20250514"),
/// ));
/// let agent = AgentRunner::builder(provider)
///     .system_prompt("You are helpful.")
///     .build()?;
/// # let _ = agent;
/// # Ok(()) }
/// ```
pub fn builder(provider: Arc<P>) -> AgentRunnerBuilder<P> { ... }
```

Use `# ` lines to hide setup boilerplate from the rendered docs while keeping the doctest compilable.

- [ ] **Step 4.1: Add doctest on `AgentRunner::builder`.**

Locate the `pub fn builder` impl on `AgentRunner` (likely in `crates/heartbit-core/src/agent/builder.rs` or `crates/heartbit-core/src/agent/mod.rs`). Add the doctest above its `pub fn builder` line, using the pattern above.

Run `cargo test --doc -p heartbit-core agent::` and verify the new doctest passes.

- [ ] **Step 4.2: Add doctest on `Tool` trait.**

In `crates/heartbit-core/src/tool/mod.rs`, find `pub trait Tool`. Above it, add:

```rust
/// A tool the agent can call.
///
/// # Example
///
/// ```rust,no_run
/// use heartbit_core::tool::{Tool, ToolOutput};
/// use heartbit_core::Error;
/// use serde_json::Value;
/// use std::future::Future;
/// use std::pin::Pin;
///
/// struct EchoTool;
///
/// impl Tool for EchoTool {
///     fn definition(&self) -> heartbit_core::ToolDefinition {
///         heartbit_core::ToolDefinition {
///             name: "echo".into(),
///             description: "Return the input unchanged.".into(),
///             input_schema: serde_json::json!({"type":"object","properties":{"msg":{"type":"string"}},"required":["msg"]}),
///         }
///     }
///
///     fn execute(
///         &self,
///         input: Value,
///     ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
///         Box::pin(async move {
///             let msg = input.get("msg").and_then(|v| v.as_str()).unwrap_or("");
///             Ok(ToolOutput::success(format!("echo: {msg}")))
///         })
///     }
/// }
/// ```
pub trait Tool: Send + Sync { ... }
```

Adapt field names if `ToolDefinition` differs from this shape (read the struct definition to confirm).

Run `cargo test --doc -p heartbit-core tool::` to verify.

- [ ] **Step 4.3: Add doctest on `Memory` trait.**

In `crates/heartbit-core/src/memory/mod.rs`, above `pub trait Memory`, add:

```rust
/// In-process or persistent memory for an agent.
///
/// # Example
///
/// ```rust,no_run
/// # async fn run() -> Result<(), heartbit_core::Error> {
/// use heartbit_core::memory::{InMemoryStore, Memory, MemoryEntry};
///
/// let store = InMemoryStore::new();
/// let id = store.store(MemoryEntry::new(
///     "agent",
///     "user",
///     "The user prefers concise responses.",
/// )).await?;
/// let entries = store.recall("agent", "preferences", 5).await?;
/// assert!(entries.iter().any(|e| e.id == id));
/// # Ok(()) }
/// ```
pub trait Memory: Send + Sync { ... }
```

Adapt to the actual `MemoryEntry::new` signature — it might take `(agent_name, user_id, content)` in a different order, or take additional fields.

Run `cargo test --doc -p heartbit-core memory::` to verify.

- [ ] **Step 4.4: Add doctest on `Guardrail` trait.**

In `crates/heartbit-core/src/agent/guardrail.rs`, above `pub trait Guardrail`, add:

```rust
/// A pre/post hook that observes or denies LLM and tool activity.
///
/// # Example
///
/// ```rust,no_run
/// use heartbit_core::agent::guardrail::{GuardAction, Guardrail};
/// use heartbit_core::llm::types::CompletionResponse;
/// use heartbit_core::Error;
/// use std::future::Future;
/// use std::pin::Pin;
///
/// struct DenyEmptyResponse;
///
/// impl Guardrail for DenyEmptyResponse {
///     fn post_llm(
///         &self,
///         response: &CompletionResponse,
///     ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
///         let is_empty = response.content.iter().all(|b| match b {
///             heartbit_core::llm::types::ContentBlock::Text { text } => text.trim().is_empty(),
///             _ => true,
///         });
///         Box::pin(async move {
///             if is_empty {
///                 Ok(GuardAction::Deny { reason: "empty response".into() })
///             } else {
///                 Ok(GuardAction::Allow)
///             }
///         })
///     }
/// }
/// ```
pub trait Guardrail: Send + Sync { ... }
```

Run `cargo test --doc -p heartbit-core agent::guardrail` to verify.

- [ ] **Step 4.5: Add doctest on `EvalRunner::new`.**

In `crates/heartbit-core/src/eval/mod.rs`, above `impl EvalRunner` or `pub fn new`, add:

```rust
/// Create an evaluation runner.
///
/// # Example
///
/// ```rust,no_run
/// # async fn run() -> Result<(), heartbit_core::Error> {
/// use heartbit_core::eval::{EvalCase, EvalRunner};
///
/// let cases = vec![
///     EvalCase::new("hello", "say hi"),
///     EvalCase::new("math", "what is 2+2?"),
/// ];
/// let runner = EvalRunner::new(cases);
/// # let _ = runner;
/// # Ok(()) }
/// ```
pub fn new(cases: Vec<EvalCase>) -> Self { ... }
```

Adapt `EvalCase::new` signature to the real one — it may take more fields like `expected_output`.

- [ ] **Step 4.6: Add doctests on workflow agent constructors.**

In `crates/heartbit-core/src/agent/workflow.rs`, add doctests on `SequentialAgent::builder` and `VotingAgent::new`:

```rust
/// Build a sequential pipeline of agents.
///
/// # Example
///
/// ```rust,no_run
/// # async fn run() -> Result<(), heartbit_core::Error> {
/// # use std::sync::Arc;
/// # use heartbit_core::{AgentRunner, AnthropicProvider, BoxedProvider, SequentialAgent};
/// # let provider = Arc::new(BoxedProvider::new(
/// #     AnthropicProvider::new("sk-...", "claude-sonnet-4-20250514"),
/// # ));
/// let a = AgentRunner::builder(provider.clone()).system_prompt("Step 1").build()?;
/// let b = AgentRunner::builder(provider).system_prompt("Step 2").build()?;
/// let pipeline = SequentialAgent::builder()
///     .agent(Box::new(a))
///     .agent(Box::new(b))
///     .build()?;
/// # let _ = pipeline;
/// # Ok(()) }
/// ```
pub fn builder() -> SequentialAgentBuilder { ... }
```

Apply the same pattern to `VotingAgent::new` (or `::builder`, whichever it has). Read the actual API first; adapt the doctest to match.

- [ ] **Step 4.7: Run all doctests.**

```bash
cargo test --doc -p heartbit-core 2>&1 | tail -10
```

Expected: all doctests pass; failure count is 0.

- [ ] **Step 4.8: Run full gate.**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --no-run 2>&1 | tail -3
```

- [ ] **Step 4.9: Commit.**

```bash
git add -u
git commit -m "docs(rustdoc): add doctests on AgentRunner, Tool, Memory, Guardrail, EvalRunner, workflow agents

6-8 doctests on the main public-API entry points. Each compiles via
cargo test --doc -p heartbit-core; cross-referenced from the upcoming
book chapters."
```

---

## Task 5: Chapter 1 — Introduction

**Files:**
- Modify: `book/src/introduction.md`

**Length target:** ~600 words. No code blocks.

**Required structure:**

```markdown
# Introduction

## What is heartbit-core

(2-3 paragraphs: Rust agentic framework. ReAct loop, parallel tool execution
via tokio::JoinSet, type-safe LLM provider abstraction. Production-grade,
not a research demo. Built around AgentRunner + Tool + Memory + Guardrail
trait abstractions.)

## What you can build with it

(Bullet list, ~5 items: chat agents, code-aware agents, multi-agent
research workflows, eval-driven prompt iteration, MCP server integrations,
durable workflows via Restate, multi-tenant SaaS — link forward to the
relevant chapter for each.)

## How this book is organized

(Brief tour of the 12 chapters: Getting Started → Agents → Tools → Memory
→ Guardrails → Workflow Agents → Orchestration → Configuration → Eval →
Recipes → Production. Reading order: the first 5 chapters in sequence;
the rest can be skimmed by need.)

## Why Rust

(One paragraph: type safety for tool I/O, async runtime that scales to
1000s of concurrent agents, single-binary deployment, no GC pauses.)

## Prerequisites

(Bullet list: Rust 1.95+, an LLM API key (Anthropic / OpenRouter / Gemini
/ OpenAI-compatible), basic familiarity with async Rust.)
```

**Required cross-links:**
- API reference: `[heartbit-core on docs.rs](https://docs.rs/heartbit-core)` once in the body.
- Source: `[heartbit on GitHub](https://github.com/heartbit-ai/heartbit)` once in the body.

- [ ] **Step 5.1: Replace placeholder content.**

Open `book/src/introduction.md` and replace the placeholder with content matching the structure above. Aim for ~600 words. Keep prose tight; no marketing fluff.

- [ ] **Step 5.2: Build and verify.**

```bash
cd book && mdbook build && cd ..
```
Expected: zero errors, zero warnings (linkcheck included).

- [ ] **Step 5.3: Commit.**

```bash
git add book/src/introduction.md
git commit -m "docs(book): chapter 1 — Introduction"
```

---

## Task 6: Chapter 2 — Getting Started

**Files:**
- Modify: `book/src/getting-started/README.md` (section landing page)
- Modify: `book/src/getting-started/installation.md`
- Modify: `book/src/getting-started/hello-agent.md`
- Modify: `book/src/getting-started/providers.md`
- Modify: `book/src/getting-started/env.md`

**Length target:** ~800 words across the 5 files combined.

### 6a — `getting-started/README.md` (~150 words)

Brief section landing. Tells the reader what they'll learn here and what they'll have at the end. Lists the 4 sub-pages.

### 6b — `getting-started/installation.md` (~100 words)

Installation steps:
- `cargo add heartbit-core` — the framework only
- `cargo add heartbit` (umbrella) — when they need Postgres / Telegram / Discord / Slack / etc.
- Mention Rust version requirement (1.95+).
- Cross-link to `[crates.io](https://crates.io/crates/heartbit-core)`.

### 6c — `getting-started/hello-agent.md` (~250 words)

The 30-line hello-world. Required content:

```markdown
A minimal agent that calls Anthropic's Claude:

\`\`\`rust,no_run
{{#include ../../../crates/heartbit-core/examples/hello_agent.rs}}
\`\`\`

Run it:

\`\`\`bash
ANTHROPIC_API_KEY=sk-... cargo run -p heartbit-core --example hello_agent
\`\`\`
```

Then 2-3 paragraphs walking through:
- What `BoxedProvider`, `RetryingProvider`, `AnthropicProvider` do (one sentence each).
- What `AgentRunner::builder().system_prompt(...).build()?` does.
- What `agent.execute(prompt).await?` returns (an `AgentOutput` with `result`, `tokens_used`, etc. — link to docs.rs).

### 6d — `getting-started/providers.md` (~200 words)

Choosing among the 4 LLM providers. Brief comparison table (markdown table):

| Provider | Strengths | Caveats |
|---|---|---|
| Anthropic | Native prompt caching, tool use mature | Higher latency than some |
| OpenRouter | One API key for ~100 models | Quality varies by underlying model |
| Gemini | Strong on long-context tasks | Tool-call format quirks |
| OpenAI-compatible | Works with local LLMs (vLLM, ollama, etc.) | YMMV on tool reliability |

Code snippet showing how to swap providers in the hello example.

Cross-link to `[heartbit_core::llm](https://docs.rs/heartbit-core/latest/heartbit_core/llm/index.html)`.

### 6e — `getting-started/env.md` (~200 words)

API keys + environment setup. Topics:
- Setting `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY` env vars.
- Using `dotenvy` or `.env` files in development (one-paragraph mention).
- For production, recommend the `vault` feature in the umbrella crate (one-sentence pointer to umbrella). Don't go deep — vault is platform-side.
- Brief mention of `HEARTBIT_ALLOW_PRIVATE_IPS=1` for the WebFetch SSRF override (link to the Tools chapter for context).

- [ ] **Step 6.1: Write `book/src/getting-started/README.md`** (~150 words). Section intro.

- [ ] **Step 6.2: Write `book/src/getting-started/installation.md`** (~100 words).

- [ ] **Step 6.3: Write `book/src/getting-started/hello-agent.md`** (~250 words). Use the `{{#include}}` block exactly as specified above.

- [ ] **Step 6.4: Write `book/src/getting-started/providers.md`** (~200 words). Include the comparison table above.

- [ ] **Step 6.5: Write `book/src/getting-started/env.md`** (~200 words).

- [ ] **Step 6.6: Build and verify.**

```bash
cd book && mdbook build && cd ..
```

Expected: clean build, no linkcheck errors.

- [ ] **Step 6.7: Commit.**

```bash
git add book/src/getting-started/
git commit -m "docs(book): chapter 2 — Getting Started"
```

---

## Task 7: Chapter 3 — Agents

**Files:**
- Modify: `book/src/agents/README.md`

**Length target:** ~900 words.

**Required structure (section headings, in order):**

```markdown
# Agents

## The ReAct loop

(2-3 paragraphs: how AgentRunner alternates between LLM calls and tool
execution. tokio::JoinSet for parallel tool calls. The loop exits on
final-text response or max_turns / max_tokens / error.)

## Building an agent

(Code: the {{#include}} block below.)

\`\`\`rust,no_run
{{#include ../../../crates/heartbit-core/examples/simple_agent.rs}}
\`\`\`

(2 paragraphs walking through the AgentRunnerBuilder calls in the example.)

## System prompts

(1 paragraph: what the system prompt does, conventions, link to the
Templates section in the Configuration chapter for the 15 built-in
templates.)

## Token budgets and turn limits

(1-2 paragraphs: max_turns (default 10), max_tokens (input cap),
max_total_tokens (lifetime cap across all turns), and what happens when
each is exceeded. Mention partial-usage error wrapping (Error::WithPartialUsage).)

## Streaming output

(1 paragraph + small inline snippet: setting on_text callback to stream
tokens to stdout. Link to docs.rs for OnText trait.)

## Events and observability

(1-2 paragraphs: the OnEvent callback fires AgentEvent variants —
RunStarted, TurnStarted, LlmCallStarted, ToolCallStarted, etc. Link to
agent::events on docs.rs. Forward link to the Production chapter for
OpenTelemetry wiring.)

## Errors and partial usage

(1 paragraph: heartbit_core::Error variants. Error::WithPartialUsage
preserves accumulated token count even when a turn fails — useful for
cost dashboards. Link to docs.rs.)
```

**Required cross-links:**
- `[AgentRunnerBuilder](https://docs.rs/heartbit-core/latest/heartbit_core/struct.AgentRunnerBuilder.html)` once.
- `[AgentEvent](https://docs.rs/heartbit-core/latest/heartbit_core/enum.AgentEvent.html)` once.
- Forward link to Configuration chapter (`./../configuration/README.md#templates`) once.
- Forward link to Production chapter (`./../production/README.md`) once.

- [ ] **Step 7.1: Write the chapter.** Aim for ~900 words across the structure above.

- [ ] **Step 7.2: Build and verify.**

```bash
cd book && mdbook build && cd ..
```

- [ ] **Step 7.3: Commit.**

```bash
git add book/src/agents/README.md
git commit -m "docs(book): chapter 3 — Agents"
```

---

## Task 8: Chapter 4 — Tools

**Files:**
- Modify: `book/src/tools/README.md`

**Length target:** ~900 words.

**Required structure:**

```markdown
# Tools

## The Tool trait

(1-2 paragraphs: definition + execute signature. ToolOutput::success vs
ToolOutput::error semantics. Tools never panic the agent loop —
panics become ToolOutput::error feeding back to the LLM.)

## Built-in tools

(1 paragraph + bullet list of the 14+ builtin tools: read, write, edit,
bash, patch, todo, web_fetch (with SSRF defense), web_search,
image_generate, tts, twitter_post, MCP client, A2A. Each gets one
sentence describing what it does. Link to
heartbit_core::tool::builtins on docs.rs.)

## Writing your own tool

(2 paragraphs intro + the {{#include}} block:)

\`\`\`rust,no_run
{{#include ../../../crates/heartbit-core/examples/custom_tool.rs}}
\`\`\`

(1 paragraph walking through the ToolDefinition / execute pair.)

## The heartbit_tool! macro

(1 paragraph: what it generates, when to use it instead of the manual
trait impl. Small inline example. Link to heartbit-macro on docs.rs.)

## Tool input validation

(1 paragraph: how validate_tool_input checks input against the
input_schema before tool dispatch. Returns errors that the LLM can
correct on the next turn.)

## Tool approval (human-in-the-loop)

(1-2 paragraphs: setting on_approval callback to interactively approve
each tool call. ApprovalDecision::{Allow, Deny, AlwaysAllow,
AlwaysDeny}. Useful for high-stakes tools.)

## MCP integration

(1 paragraph: heartbit-core ships with a production-grade MCP client.
Connect with `McpClient::connect`, point an agent at it, and the MCP
server's tools become available alongside built-ins. Forward link to
the MCP recipe.)
```

**Required cross-links:**
- `[heartbit_core::tool::builtins](https://docs.rs/heartbit-core/latest/heartbit_core/tool/builtins/index.html)`
- `[McpClient](https://docs.rs/heartbit-core/latest/heartbit_core/tool/struct.McpClient.html)`
- Forward link to `./../recipes/mcp-integration.md`

- [ ] **Step 8.1: Write the chapter** (~900 words).

- [ ] **Step 8.2: Build, verify, commit.**

```bash
cd book && mdbook build && cd ..
git add book/src/tools/README.md
git commit -m "docs(book): chapter 4 — Tools"
```

---

## Task 9: Chapter 5 — Memory

**Files:**
- Modify: `book/src/memory/README.md`

**Length target:** ~800 words.

**Required structure:**

```markdown
# Memory

## The Memory trait

(1-2 paragraphs: 6 trait methods — store, recall, update, forget,
add_link, prune. Returns futures, async-safe.)

## InMemoryStore and NamespacedMemory

(1 paragraph each. InMemoryStore: thread-safe HashMap-backed.
NamespacedMemory: wraps any Memory impl with tenant/agent prefix
isolation.)

## Memory entries

(1-2 paragraphs: MemoryEntry fields — content, memory_type
(Episodic/Semantic/Reflection), keywords, summary, strength,
related_ids, source_ids, author info. Strength decays over time
(Ebbinghaus); reinforced on access.)

## The 5 memory tools

(1 paragraph + bullet list: agents that have a Memory wired in get 5
tools — store, recall, update, forget, consolidate — through the
MemGPT pattern. Brief description of each.)

## Recall and ranking

(1 paragraph: BM25 keyword scoring (2x boost on keyword matches) +
composite scoring across recency, importance, relevance, strength.
Link to docs.rs.)

## Embeddings and hybrid search

(1-2 paragraphs: trait abstraction. Local-embedding feature in the
umbrella crate uses fastembed. Hybrid search combines BM25 with
cosine similarity (RRF). Forward link to umbrella crate.)

## Memory lifecycle

(1 paragraph + the {{#include}} block:)

\`\`\`rust,no_run
{{#include ../../../crates/heartbit-core/examples/memory.rs}}
\`\`\`

## Postgres-backed memory

(1 paragraph: gated behind the postgres feature in the heartbit
umbrella crate. Schema auto-migrates on first run. Forward link to the
Production chapter and umbrella crate docs.)
```

**Required cross-links:**
- `[heartbit_core::memory::InMemoryStore](https://docs.rs/heartbit-core/latest/heartbit_core/memory/struct.InMemoryStore.html)`
- `[heartbit_core::memory::NamespacedMemory](https://docs.rs/heartbit-core/latest/heartbit_core/memory/struct.NamespacedMemory.html)`
- Forward link to `./../production/README.md`

- [ ] **Step 9.1: Write the chapter** (~800 words).

- [ ] **Step 9.2: Build, verify, commit.**

```bash
cd book && mdbook build && cd ..
git add book/src/memory/README.md
git commit -m "docs(book): chapter 5 — Memory"
```

---

## Task 10: Chapter 6 — Guardrails

**Files:**
- Modify: `book/src/guardrails/README.md`

**Length target:** ~800 words.

**Required structure:**

```markdown
# Guardrails

## The Guardrail trait

(1-2 paragraphs: 4 hooks — pre_llm, post_llm, pre_tool, post_tool.
GuardAction::{Allow, Deny, Warn, Kill}. First Deny wins across multiple
guardrails. Async, returns futures.)

## Built-in guardrails

(One subsection per built-in, ~50 words each:)

### LLM judge
### Secret scanner
### PII guardrail
### Content fence
### Action budget
### Behavioral monitor
### Tool policy
### Injection classifier
### Sensor security (umbrella, sensor feature)

## Composing multiple guardrails

(1 paragraph + small inline snippet showing AgentRunnerBuilder::guardrails(vec![...]).)

## Example: full guardrail stack

(The {{#include}} block:)

\`\`\`rust,no_run
{{#include ../../../crates/heartbit-core/examples/guardrails.rs}}
\`\`\`

## When to write your own

(1 paragraph + signature pattern. Link to the Tools recipe for the
custom_tool pattern; guardrail authoring is similar but on the post_tool
hook.)
```

**Required cross-links:**
- `[heartbit_core::agent::guardrails](https://docs.rs/heartbit-core/latest/heartbit_core/agent/guardrails/index.html)` once.

- [ ] **Step 10.1: Write the chapter** (~800 words).

- [ ] **Step 10.2: Build, verify, commit.**

```bash
cd book && mdbook build && cd ..
git add book/src/guardrails/README.md
git commit -m "docs(book): chapter 6 — Guardrails"
```

---

## Task 11: Chapter 7 — Workflow Agents

**Files:**
- Modify: `book/src/workflow-agents/README.md`

**Length target:** ~700 words.

**Required structure:**

```markdown
# Workflow Agents

## When to use a workflow vs a single agent

(2 paragraphs: deterministic orchestration without LLM cost for the
dispatcher. Use a single AgentRunner when one LLM call is enough; use
workflow agents when the steps are known in advance.)

## Sequential

(1 paragraph + the {{#include}} block:)

\`\`\`rust,no_run
{{#include ../../../crates/heartbit-core/examples/sequential_agent.rs}}
\`\`\`

## Parallel

(1 paragraph: ParallelAgent runs all sub-agents concurrently via
tokio::JoinSet, gathers results in deterministic order. Inline snippet
showing the builder pattern.)

## Loop

(1 paragraph: LoopAgent repeats a single sub-agent until should_stop
returns true or max_iterations is reached. Useful for refinement loops.
Inline snippet.)

## DAG

(1 paragraph + the {{#include}} block:)

\`\`\`rust,no_run
{{#include ../../../crates/heartbit-core/examples/dag_agent.rs}}
\`\`\`

(1 paragraph: petgraph-backed BFS execution; topological order respected;
parallelism within a tier.)

## Voting / Debate / Mixture-of-agents

(1 paragraph each, brief. Voting: majority consensus. Debate: multi-round
back-and-forth between agents with different prompts. MoA: proposers +
synthesizer. Link to docs.rs for each.)

## Choosing the right pattern

(1 paragraph: a small decision-tree. "Need deterministic order? Sequential.
Independent fan-out? Parallel. Refinement until quality? Loop. Conditional
branching? DAG. Multiple opinions? Voting/Debate/MoA.")
```

**Required cross-links:**
- `[heartbit_core::SequentialAgent](https://docs.rs/heartbit-core/latest/heartbit_core/struct.SequentialAgent.html)`, `[ParallelAgent](https://docs.rs/heartbit-core/latest/heartbit_core/struct.ParallelAgent.html)`, `[LoopAgent](https://docs.rs/heartbit-core/latest/heartbit_core/struct.LoopAgent.html)`, `[DagAgent](https://docs.rs/heartbit-core/latest/heartbit_core/struct.DagAgent.html)`, `[VotingAgent](https://docs.rs/heartbit-core/latest/heartbit_core/struct.VotingAgent.html)`, `[DebateAgent](https://docs.rs/heartbit-core/latest/heartbit_core/struct.DebateAgent.html)`, `[MixtureOfAgentsAgent](https://docs.rs/heartbit-core/latest/heartbit_core/struct.MixtureOfAgentsAgent.html)`.

- [ ] **Step 11.1: Write the chapter** (~700 words).

- [ ] **Step 11.2: Build, verify, commit.**

```bash
cd book && mdbook build && cd ..
git add book/src/workflow-agents/README.md
git commit -m "docs(book): chapter 7 — Workflow Agents"
```

---

## Task 12: Chapter 8 — Multi-Agent Orchestration

**Files:**
- Modify: `book/src/orchestration/README.md`

**Length target:** ~700 words.

**Required structure:**

```markdown
# Multi-Agent Orchestration

## When to use the Orchestrator

(2 paragraphs: dynamic, LLM-driven dispatch when sub-agents are picked at
runtime. Versus workflow agents (deterministic structure). Trade-off:
flexibility vs predictability + cost.)

## The Orchestrator and OrchestratorBuilder

(1 paragraph: building an orchestrator with multiple SubAgentConfig
entries. Single-agent fast path: when agents.len() == 1, the orchestrator
bypasses dispatch and runs the agent directly.)

## Sub-agent dispatch tools

(One paragraph each:)

### DelegateTaskTool
(Picks one named sub-agent and runs it.)

### FormSquadTool
(Spawns a dynamic squad: composes a sub-team for a particular task,
runs them in parallel, returns aggregated result.)

### SpawnAgentTool
(Spawns a sub-agent with custom system prompt + tools, on the fly.)

## Example

(The {{#include}} block:)

\`\`\`rust,no_run
{{#include ../../../crates/heartbit-core/examples/multi_agent.rs}}
\`\`\`

## Blackboard for shared state

(1 paragraph: blackboard pattern for sub-agents to read/write shared
data via blackboard_read, blackboard_write, blackboard_list tools.
Useful for FormSquad coordination.)
```

**Required cross-links:**
- `[heartbit_core::Orchestrator](https://docs.rs/heartbit-core/latest/heartbit_core/struct.Orchestrator.html)`
- `[heartbit_core::OrchestratorBuilder](https://docs.rs/heartbit-core/latest/heartbit_core/struct.OrchestratorBuilder.html)`

- [ ] **Step 12.1: Write the chapter** (~700 words).

- [ ] **Step 12.2: Build, verify, commit.**

```bash
cd book && mdbook build && cd ..
git add book/src/orchestration/README.md
git commit -m "docs(book): chapter 8 — Multi-Agent Orchestration"
```

---

## Task 13: Chapter 9 — Configuration

**Files:**
- Modify: `book/src/configuration/README.md`

**Length target:** ~600 words.

**Required structure:**

```markdown
# Configuration

## HeartbitConfig from TOML

(1 paragraph + the {{#include}} block:)

\`\`\`rust,no_run
{{#include ../../../crates/heartbit-core/examples/from_toml.rs}}
\`\`\`

(1 paragraph walking through the TOML structure.)

## Provider configuration

(1 paragraph: provider section. provider_type, model, base_url,
prompt_caching, plus the cascade and retry sub-sections.)

## Cascade and retry

(1 paragraph: CascadeConfig (try cheap → fall back to expensive),
RetryProviderConfig (exponential backoff for transient failures).)

## Per-agent overrides

(1 paragraph: each [[agents]] entry can override the global provider,
guardrails, memory, etc. Brief inline TOML snippet showing per-agent
provider override.)

## Templates

(1 paragraph: 15 built-in templates ship with heartbit-core. Reference by
name in agents config: template = "researcher". List the 15 names. Link
to the templates dir on GitHub.)

## Skills

(1 paragraph: 10 built-in skill packs (rust-expert, python-expert,
typescript-expert, sql-expert, git-expert, docker, kubernetes,
api-design, security, testing). Auto-injected based on task keywords.
Link to skills dir on GitHub.)

## MCP server presets

(1 paragraph: 10 built-in MCP server presets — github, gitlab, slack,
notion, postgresql, brave-search, sentry, linear, google-calendar, jira.
Reference by name; resolves to preset config.)
```

**Required cross-links:**
- `[heartbit_core::HeartbitConfig](https://docs.rs/heartbit-core/latest/heartbit_core/struct.HeartbitConfig.html)`
- `[crates/heartbit-core/templates/](https://github.com/heartbit-ai/heartbit/tree/main/crates/heartbit-core/templates)` for templates list
- `[crates/heartbit-core/skills/](https://github.com/heartbit-ai/heartbit/tree/main/crates/heartbit-core/skills)` for skills list
- `[crates/heartbit-core/mcp-presets/](https://github.com/heartbit-ai/heartbit/tree/main/crates/heartbit-core/mcp-presets)` for MCP presets list

- [ ] **Step 13.1: Write the chapter** (~600 words).

- [ ] **Step 13.2: Build, verify, commit.**

```bash
cd book && mdbook build && cd ..
git add book/src/configuration/README.md
git commit -m "docs(book): chapter 9 — Configuration"
```

---

## Task 14: Chapter 10 — Eval Framework

**Files:**
- Modify: `book/src/eval/README.md`

**Length target:** ~500 words.

**Required structure:**

```markdown
# Eval Framework

## Why eval

(1 paragraph: agents are LLM apps; LLMs drift. Eval framework lets you
codify "this prompt + this tool stack should do X for these inputs" and
catch regressions in CI.)

## EvalRunner and EvalCase

(1 paragraph + the {{#include}} block:)

\`\`\`rust,no_run
{{#include ../../../crates/heartbit-core/examples/eval.rs}}
\`\`\`

## Built-in scorers

(1 paragraph + bullet list of 7 scorers:)

- TrajectoryScorer: matches expected sequence of tool calls
- KeywordScorer: required/forbidden keywords in final output
- SimilarityScorer: cosine similarity of output to reference
- CostScorer: tokens within budget
- LatencyScorer: completed under wall-clock cap
- ToolCallCountScorer: tool calls within bounds
- SafetyScorer: guardrail-pass rate

(One sentence per item.)

## Writing custom scorers

(1 paragraph + small inline trait-impl snippet. Scorer trait + return
ScoreReport. Composable.)

## Running evals in CI

(1 paragraph: cargo test --doc + an eval-binary in your project that
runs cases and emits machine-readable report. Forward link to the
"Eval-driven prompt iteration" recipe.)
```

**Required cross-links:**
- `[heartbit_core::eval::EvalRunner](https://docs.rs/heartbit-core/latest/heartbit_core/eval/struct.EvalRunner.html)`
- `[heartbit_core::eval::EvalCase](https://docs.rs/heartbit-core/latest/heartbit_core/eval/struct.EvalCase.html)`
- Forward link to `./../recipes/eval-driven.md`

- [ ] **Step 14.1: Write the chapter** (~500 words).

- [ ] **Step 14.2: Build, verify, commit.**

```bash
cd book && mdbook build && cd ..
git add book/src/eval/README.md
git commit -m "docs(book): chapter 10 — Eval Framework"
```

---

## Task 15: Chapter 11 — Recipes (cookbook)

**Files:**
- Modify: `book/src/recipes/README.md` (cookbook landing page)
- Modify: `book/src/recipes/chat-with-search.md`
- Modify: `book/src/recipes/code-aware.md`
- Modify: `book/src/recipes/multi-agent-research.md`
- Modify: `book/src/recipes/persistent-memory.md`
- Modify: `book/src/recipes/eval-driven.md`
- Modify: `book/src/recipes/mcp-integration.md`

**Length target:** ~1,200 words across all 7 files (cookbook landing + 6 recipes).

### 15a — Cookbook landing (`recipes/README.md`, ~150 words)

Brief intro: what the cookbook is, how to use it, brief description of each of the 6 recipes with hyperlinks.

### 15b — Recipe: Chat agent with web search (~200 words)

Build a conversational agent that uses `web_search` + `web_fetch`. Show:
- builder() with `.tool(WebSearchTool)` and `.tool(WebFetchTool)`
- on_text streaming for live UX
- Brief discussion of the SSRF defense in WebFetch (link to Tools chapter)

Use the `{{#include}}` directive against `crates/heartbit-core/examples/simple_agent.rs` since it already has these tools wired (or write the snippet inline if the example doesn't fit; ~20 LOC inline is OK in cookbook entries).

### 15c — Recipe: Code-aware agent (~200 words)

Agent with file tools (read, write, edit, patch) + LSP integration. Show wiring up an LspManager from the umbrella crate. Cross-link to the umbrella's `lsp` module rustdoc.

### 15d — Recipe: Multi-agent research workflow (~200 words)

Combine `multi_agent.rs` example with a SequentialAgent or DagAgent
pattern. Show: orchestrator with researcher + writer sub-agents, output piped through.

Use `{{#include}}` against `crates/heartbit-core/examples/multi_agent.rs`.

### 15e — Recipe: Long-running agent with persistent memory (~200 words)

Setting up `PostgresMemoryStore` (umbrella crate, postgres feature). Show: provisioning Postgres, the connection string, how the schema auto-migrates. Cross-link to the heartbit umbrella crate.

### 15f — Recipe: Eval-driven prompt iteration (~250 words)

Workflow: write 5–10 EvalCases, run EvalRunner against your agent, iterate the prompt until scores hit the threshold. Show: a small CI integration that fails the build if regression is detected.

Use `{{#include}}` against `crates/heartbit-core/examples/eval.rs`.

### 15g — Recipe: MCP server integration (~200 words)

Connecting an MCP server (e.g., GitHub MCP). Show: `McpClient::connect`, passing the client to AgentRunnerBuilder, the agent now has all of GitHub MCP's tools alongside built-ins.

Use `{{#include}}` against `crates/heartbit-core/examples/mcp_agent.rs`.

- [ ] **Step 15.1: Write `recipes/README.md`** (~150 words).
- [ ] **Step 15.2: Write `recipes/chat-with-search.md`** (~200 words).
- [ ] **Step 15.3: Write `recipes/code-aware.md`** (~200 words).
- [ ] **Step 15.4: Write `recipes/multi-agent-research.md`** (~200 words).
- [ ] **Step 15.5: Write `recipes/persistent-memory.md`** (~200 words).
- [ ] **Step 15.6: Write `recipes/eval-driven.md`** (~250 words).
- [ ] **Step 15.7: Write `recipes/mcp-integration.md`** (~200 words).

- [ ] **Step 15.8: Build and verify.**

```bash
cd book && mdbook build && cd ..
```

- [ ] **Step 15.9: Commit.**

```bash
git add book/src/recipes/
git commit -m "docs(book): chapter 11 — Recipes (cookbook)"
```

---

## Task 16: Chapter 12 — Production Considerations

**Files:**
- Modify: `book/src/production/README.md`

**Length target:** ~400 words.

**Required structure:**

```markdown
# Production Considerations

## Sandboxing

(1-2 paragraphs: Linux landlock via the heartbit umbrella crate's
`sandbox` feature. BashTool::with_sandbox_policy() applies it. Forward
link to the umbrella crate.)

## Resource limits

(1-2 paragraphs: max_turns, max_tokens, max_total_tokens,
max_identical_tool_calls (doom loop), run_timeout. Recommended caps
per agent in production. Link to AgentRunnerBuilder rustdoc.)

## Observability

(1-2 paragraphs: events via OnEvent, OpenTelemetry via the heartbit-cli
binary's setup_telemetry, Prometheus metrics in daemon mode. Forward
link to docs/platform.md for the full setup guide.)

## Multi-tenancy

(1 paragraph: NamespacedMemory, JWT validation via the umbrella's
auth/jwt module. Forward link to crates/heartbit-cli/README.md for
deployment.)

## Going beyond library mode

(1 paragraph: the heartbit umbrella crate's daemon feature gives you
multi-tenant Kafka-backed runtime, Postgres task store, dashboard, SSE
events. Pointer to docs/platform.md.)
```

**Required cross-links:**
- `[docs/platform.md](https://github.com/heartbit-ai/heartbit/blob/main/docs/platform.md)`
- `[crates/heartbit-cli/README.md](https://github.com/heartbit-ai/heartbit/blob/main/crates/heartbit-cli/README.md)`
- `[heartbit umbrella crate](https://crates.io/crates/heartbit)`

- [ ] **Step 16.1: Write the chapter** (~400 words).

- [ ] **Step 16.2: Build, verify, commit.**

```bash
cd book && mdbook build && cd ..
git add book/src/production/README.md
git commit -m "docs(book): chapter 12 — Production Considerations"
```

---

## Task 17: Polish pass

**Files:** none modified (verification only; spot edits if needed).

- [ ] **Step 17.1: Run `mdbook serve` locally and click through every page.**

```bash
cd book && mdbook serve --port 3000
```

Open `http://localhost:3000/` in a browser. Walk through:
- Introduction → Getting Started → Agents → Tools → Memory → Guardrails → Workflow Agents → Orchestration → Configuration → Eval → Recipes → Production.

Look for:
- Tone shifts (early chapters too terse, later chapters too verbose, etc.).
- Missing cross-links where one would help the reader.
- Code blocks that don't render (mdBook syntax errors).
- Headings that are inconsistent (some use Title Case, some sentence case — pick one).

- [ ] **Step 17.2: Run linkcheck explicitly.**

```bash
cd book && mdbook build 2>&1 | grep -i "error\|warning" | head -20
```

Expected: zero error / warning lines from linkcheck. If any, fix the dead links and rebuild.

- [ ] **Step 17.3: Verify all `{{#include}}` paths resolve.**

```bash
cd book && mdbook build 2>&1 | grep -i "include" | head
```

Expected: zero errors about missing include files.

- [ ] **Step 17.4: Spot-check a few key chapters via Markdown grep.**

```bash
# Every chapter should have at least one docs.rs link:
grep -L "docs.rs/heartbit-core" book/src/agents/README.md book/src/tools/README.md \
    book/src/memory/README.md book/src/guardrails/README.md \
    book/src/workflow-agents/README.md book/src/orchestration/README.md \
    book/src/configuration/README.md book/src/eval/README.md
```

Expected: empty output (every chapter has at least one docs.rs link). If any chapter is missing a docs.rs link, add one.

- [ ] **Step 17.5: Commit any polish edits.**

If edits were needed:

```bash
git add book/
git commit -m "docs(book): polish pass — fix linkcheck issues, tone, missing cross-links"
```

If nothing changed, skip this commit.

---

## Task 18: One-line CONTRIBUTING.md update

**Files:**
- Modify: `CONTRIBUTING.md`

- [ ] **Step 18.1: Add the maintenance note.**

Read `CONTRIBUTING.md`. Find a sensible spot (likely under a "Documentation" or "Pull Requests" subsection; add the subsection if missing). Add:

```markdown
## Documentation

If you change a public API in `heartbit-core`, update the relevant
chapter in `book/src/`. The book is published at
[docs.heartbit.ai](https://docs.heartbit.ai) and is the canonical
narrative reference for library users.
```

- [ ] **Step 18.2: Commit.**

```bash
git add CONTRIBUTING.md
git commit -m "docs(contributing): note book maintenance for public API changes"
```

---

## Task 19: Add docs badge to top-level README

**Files:**
- Modify: `README.md`

- [ ] **Step 19.1: Add the docs badge.**

In `README.md`, find the existing badge row (crates.io / docs.rs / CI / license). Add this badge to it:

```markdown
[![Book](https://img.shields.io/badge/book-docs.heartbit.ai-blue)](https://docs.heartbit.ai)
```

Place it immediately after the docs.rs badge.

- [ ] **Step 19.2: Verify the badge URL is sensible.**

```bash
grep "docs.heartbit.ai" README.md | head
```

Expected: the badge line and the link target.

- [ ] **Step 19.3: Commit.**

```bash
git add README.md
git commit -m "docs: add docs.heartbit.ai badge to top-level README"
```

---

## Task 20: DNS + Pages config (manual, by project owner)

**This task is performed by the project owner outside the code work.** It's listed here only so the implementer knows the round is incomplete without it.

- [ ] **Step 20.1: Configure DNS at the registrar for `heartbit.ai`.**

Add a CNAME record:

```
Type:   CNAME
Name:   docs
Target: heartbit-ai.github.io.
TTL:    3600  (or registrar default)
```

Wait for propagation (5 min – 24 h, usually < 30 min).

- [ ] **Step 20.2: Verify DNS resolves.**

```bash
dig docs.heartbit.ai +short
```

Expected: a `heartbit-ai.github.io.` CNAME entry pointing to GitHub's IPs.

- [ ] **Step 20.3: Configure GitHub Pages in repo settings.**

In the GitHub web UI: Settings → Pages.

- Source: `gh-pages` branch, root (`/`)
- Custom domain: `docs.heartbit.ai`
- Click "Save"
- After GitHub validates DNS, tick "Enforce HTTPS"

- [ ] **Step 20.4: Verify the site is live.**

```bash
curl -sI https://docs.heartbit.ai/ | grep -i "^http"
```

Expected: `HTTP/2 200`.

Visit https://docs.heartbit.ai in a browser. The book renders. Click through a few chapters. Search works.

---

## Self-Review

Run after Task 19 completes (Task 20 is manual; not part of code review).

- [ ] **Step S.1: Spec coverage.**

Verify each spec section maps to a task above:

- Architecture (book/, .github/workflows/) → Tasks 1, 2 ✓
- Hosting & domain → Task 2 (CNAME) + Task 20 (DNS) ✓
- TOC (12 chapters) → Tasks 5–16 ✓
- Content sources & generation → Tasks 3 (examples) + 4 (doctests) + chapter `{{#include}}` directives ✓
- CI & deployment → Task 2 ✓
- Tone & voice → "Chapter Writing Convention" preamble + chapter task instructions ✓
- Maintenance → Task 18 (CONTRIBUTING.md) ✓
- Sequencing & 19 tasks → mapped 1:1 (with Task 20 added for the manual DNS work) ✓

- [ ] **Step S.2: Run the full exit-criteria battery.**

```bash
# 1. mdbook build succeeds
cd book && mdbook build && cd ..

# 2. examples + doctests still pass
cargo check --examples -p heartbit-core
cargo test --doc -p heartbit-core 2>&1 | tail -3

# 3. workspace gates still green
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --no-run 2>&1 | tail -3
cargo test --workspace --lib 2>&1 | grep "test result"
```

All exit 0.

- [ ] **Step S.3: Verify badge + cross-links.**

```bash
grep -c "docs.heartbit.ai" README.md
grep -c "docs.heartbit.ai" CONTRIBUTING.md
```

Expected: ≥1 in each.

- [ ] **Step S.4: Verify chapter count.**

```bash
ls book/src/ | wc -l
```

Expected: 12 chapter dirs/files (introduction.md + 11 subdirs) plus SUMMARY.md = 13 entries.

---

## Out of Scope (per spec)

These are NOT part of this plan and should NOT be addressed in this round:

- Platform/operations docs (lives in `docs/platform.md` + `crates/heartbit-cli/README.md`; book Production chapter just links to them).
- Comparison with LangChain / Mastra / Eliza.
- Translations.
- Versioned documentation.
- Replacing rustdoc with the book.
- Custom theming, branding, logos beyond default mdBook + the `CNAME` file.
- SEO beyond mdBook's default sitemap.
- Heavy interactive widgets (live REPL, runnable browser examples).

If any of these are tempting during execution: stop, note it as a follow-up, proceed with the plan as written.
