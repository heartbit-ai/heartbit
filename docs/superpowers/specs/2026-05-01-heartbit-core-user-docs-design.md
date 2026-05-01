# heartbit-core User Documentation Design

**Date:** 2026-05-01
**Status:** Design — pending user approval before implementation plan
**Scope:** Build a comprehensive long-form user guide ("the heartbit book") for `heartbit-core`, hosted at `docs.heartbit.ai` via GitHub Pages. The book lives in `book/` at the repo root, paired with rustdoc on docs.rs as the API reference.
**Estimated effort:** 5–7 days of focused writing, executed as ~19 small mergeable commits.

## Background

Following the B3 round (commits `6f12c26..3ebdea5`), `heartbit-core` is positioned as the official Rust agentic framework on crates.io. The top-level `README.md` was rewritten to lead with `cargo add heartbit-core`, and `crates/heartbit-core/README.md` is the docs.rs landing page. The B3 spec/plan and the production-readiness audit (2026-05-01) both confirmed the framework is feature-complete and production-grade for library use, with rich content surface (workflow agents, guardrails, memory system, MCP, eval framework) that warrants long-form treatment beyond what a README can carry.

Library users who land on `crates.io/crates/heartbit-core` get a quickstart and a feature list. The next questions they ask — "how do I write a custom guardrail?", "when do I use Sequential vs DAG vs Voting?", "how does the memory consolidation pipeline work?" — are not README questions; they're book questions. Currently those readers have nowhere to go except the rustdoc API reference, which is excellent for "what does this method do" but poor for "what should I be reading first" and "how do these pieces fit together."

A narrative book closes that gap. It's the convention for foundational Rust frameworks: Tokio, Axum, Tracing, Serde, Bevy, Diesel — all maintain a separate mdBook-built guide alongside their docs.rs API reference. Nothing about heartbit-core's audience or scope deviates from that pattern.

## Goals

1. Make `docs.heartbit.ai` the canonical entry point for library users learning the framework.
2. Cover every major public-API surface with a hand-written narrative chapter — agents, tools, memory, guardrails, workflow agents, orchestration, configuration, eval — at sufficient depth that a reader can build a real agent end-to-end without leaving the book.
3. Pair the book with rustdoc on docs.rs as the API reference; cross-link the two.
4. Keep examples honest: every code block in the book is sourced from a real, compiled file in `crates/heartbit-core/examples/` (via mdBook's `{{#include}}` directive) or a doctest in `crates/heartbit-core/src/`.
5. Build and deploy automatically via GitHub Actions on push to main; fail the build on dead internal links.
6. No platform/operations content in this book — that lives in `crates/heartbit-cli/README.md` and `docs/platform.md` and is linked from the book's "Production" chapter.

## Non-Goals

- **Platform/operations docs** — daemon mode, multi-tenant deployment, Kafka topology, dashboard ops. Those exist (`docs/platform.md`, `crates/heartbit-cli/README.md`) and are linked from this book's Production chapter; not duplicated.
- **Comparison with LangChain / Mastra / Eliza** — comparisons date fast and read defensive. The Introduction states what heartbit-core is, not what it isn't.
- **Translations** — English only this round. mdBook supports localization via subdirectories if added later.
- **Versioned documentation** — one version of the book lives at `docs.heartbit.ai`. mdBook supports versioning via subdirs (`/v1/`, `/v2/`) if/when needed.
- **Replacing rustdoc** — the book is narrative; the API reference is rustdoc. The book links *to* docs.rs, not *replaces* it.
- **Heavy theming / branding** — default mdBook theme, plus a `CNAME` file. Custom CSS / fonts / logos can land later.
- **SEO beyond mdBook defaults** — mdBook generates a sitemap. No additional SEO work in this round.

## Design

### Architecture

```
heartbit/                              repo root
├── book/                              NEW. mdBook source.
│   ├── book.toml                      mdBook config: title, search, mermaid, linkcheck
│   ├── src/
│   │   ├── SUMMARY.md                 chapter index (table of contents)
│   │   ├── introduction.md            ch 1: Introduction
│   │   ├── getting-started/           ch 2: Getting Started (multi-page section)
│   │   │   ├── README.md              section index
│   │   │   ├── installation.md
│   │   │   ├── hello-agent.md
│   │   │   ├── providers.md
│   │   │   └── env.md
│   │   ├── agents/                    ch 3: Agents
│   │   ├── tools/                     ch 4: Tools
│   │   ├── memory/                    ch 5: Memory
│   │   ├── guardrails/                ch 6: Guardrails
│   │   ├── workflow-agents/           ch 7: Workflow Agents
│   │   ├── orchestration/             ch 8: Multi-Agent Orchestration
│   │   ├── configuration/             ch 9: Configuration
│   │   ├── eval/                      ch 10: Eval Framework
│   │   ├── recipes/                   ch 11: Recipes (cookbook)
│   │   └── production/                ch 12: Production Considerations
│   └── theme/                         (optional; mdBook hook for CNAME, custom JS)
├── .github/workflows/book.yml         NEW. mdBook → GitHub Pages CI.
├── crates/heartbit-core/examples/     EXISTING + ~3 new files added in this round
└── crates/heartbit-core/src/          ~6–8 doctests added in this round
```

### Hosting & Domain

- **Repository setting**: GitHub Pages → source: `gh-pages` branch, `/` (root).
- **Custom domain**: `docs.heartbit.ai`, configured via `CNAME` file deployed by CI.
- **DNS**: one CNAME record at the user's `heartbit.ai` registrar:
  ```
  docs.heartbit.ai.   CNAME   heartbit-ai.github.io.
  ```
  This step is performed by the project owner outside the implementation work.
- **HTTPS**: GitHub Pages auto-provisions a Let's Encrypt cert once DNS resolves; "Enforce HTTPS" enabled in repo settings.
- **Build cmd locally**: `cd book && mdbook serve` → live preview at `http://localhost:3000`.

### Table of Contents (Locked)

```
1. Introduction
   - What is heartbit-core
   - Why Rust for agents
2. Getting Started
   - Installation (cargo add heartbit-core)
   - Hello agent (~30 LOC)
   - Choosing an LLM provider (Anthropic / OpenRouter / Gemini / OpenAI-compat)
   - API keys and environment
3. Agents
   - AgentRunner and the ReAct loop
   - System prompts and templates
   - Token budgets and turn limits
   - Streaming output
   - Events and observability
   - Errors and partial usage
4. Tools
   - The Tool trait and ToolOutput
   - Built-in tools (read, write, edit, bash, patch, todo, web_fetch, web_search, …)
   - Writing your own tool (the heartbit_tool! macro)
   - Tool input validation
   - Tool approval (human-in-the-loop)
5. Memory
   - The Memory trait
   - InMemoryStore and NamespacedMemory
   - Memory entries: types, keywords, strength, decay
   - The 5 memory tools (store, recall, update, forget, consolidate)
   - Embeddings and hybrid search (BM25 + vectors)
   - Postgres-backed memory (umbrella crate, brief)
6. Guardrails
   - The Guardrail trait — pre/post LLM and tool hooks
   - Built-in guardrails (LLM judge, secret scanner, PII, content fence,
     action budget, behavioral monitor, tool policy, injection classifier)
   - Composing multiple guardrails
   - When to write your own
7. Workflow Agents
   - Sequential, Parallel, Loop
   - DAG (with petgraph)
   - Voting, Debate, Mixture-of-agents
   - Choosing the right pattern
8. Multi-Agent Orchestration
   - Orchestrator and OrchestratorBuilder
   - Sub-agent dispatch (DelegateTask, FormSquad, SpawnAgent)
   - Blackboard for shared state
   - Single-agent fast path
9. Configuration
   - HeartbitConfig from TOML
   - Per-agent provider config (cascade, retry)
   - Templates (15 built-in)
   - Skills (10 built-in)
   - MCP server presets (10 built-in)
10. Eval Framework
    - EvalRunner and EvalCase
    - Built-in scorers (trajectory, keyword, similarity, cost, latency,
      tool-call count, safety)
    - Writing custom scorers
11. Recipes (cookbook)
    - Chat agent with web search
    - Code-aware agent (file tools + LSP)
    - Multi-agent research workflow
    - Long-running agent with persistent memory
    - Eval-driven prompt iteration
    - MCP server integration
12. Production Considerations
    - Sandboxing (Linux landlock)
    - Resource limits and budgets
    - Observability (events, OpenTelemetry, Prometheus)
    - Pointer to platform/daemon docs
       (link to docs/platform.md and crates/heartbit-cli/README.md)
```

### Content Sources & Generation Strategy

For each chapter, content combines three sources:

1. **Hand-written prose** — concepts, motivation, when-to-use, gotchas. The narrative spine. ~600–1,000 words per chapter (less for short chapters like Production).

2. **`{{#include}}` directives pulling from real example files.** mdBook syntax:
   ```markdown
   ```rust,no_run
   {{#include ../../crates/heartbit-core/examples/hello_agent.rs}}
   ```
   ```
   These compile via the existing CI gate (`cargo check --examples`).

3. **Cross-links into rustdoc on docs.rs** for full API reference. Pattern:
   ```markdown
   See [`AgentRunnerBuilder`][builder] for all available knobs.
   [builder]: https://docs.rs/heartbit-core/latest/heartbit_core/struct.AgentRunnerBuilder.html
   ```

**Per-chapter example mapping:**

| Chapter | Example file (in `crates/heartbit-core/examples/`) |
|---|---|
| Getting Started | `hello_agent.rs` (existing) |
| Agents | `simple_agent.rs` (existing) |
| Tools | `custom_tool.rs` (existing) + `mcp_agent.rs` (existing) snippets |
| Memory | `memory.rs` (existing) |
| Guardrails | `guardrails.rs` (existing) |
| Workflow Agents | `sequential_agent.rs` (NEW) + `dag_agent.rs` (NEW) |
| Orchestration | `multi_agent.rs` (existing) |
| Configuration | `from_toml.rs` (NEW) + reference to existing `heartbit.toml` |
| Eval | `eval.rs` (existing) |
| Recipes | reuses existing + 1 new recipe-specific example |

**New examples to add: 3–4 files** (`sequential_agent.rs`, `dag_agent.rs`, `from_toml.rs`, optionally one Recipes-specific). Each ~30–60 LOC.

**Doctests** added selectively to:
- `AgentRunner::builder()`
- `Tool` trait
- `Memory` trait
- `Guardrail` trait
- `EvalRunner::new()`
- The workflow agent constructors (`SequentialAgent::new`, `DagAgent::new`, `VotingAgent::new`)

~6–8 doctests of 10–15 LOC each. These compile-check on every `cargo test --doc -p heartbit-core` (already part of CI gate).

**No fictional examples.** Every code block in the book traces to either a real `examples/*.rs` file or a doctest in source. If a chapter needs a snippet that doesn't exist anywhere, write it as a real example first, then `{{#include}}` it.

### CI & Deployment

`.github/workflows/book.yml`:

```yaml
name: Book

on:
  push:
    branches: [main]
    paths: ['book/**', 'crates/heartbit-core/examples/**', '.github/workflows/book.yml']
  pull_request:
    paths: ['book/**', 'crates/heartbit-core/examples/**']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: peaceiris/actions-mdbook@v2
        with:
          mdbook-version: 'latest'
      - run: cargo install mdbook-mermaid mdbook-linkcheck
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

**Three guards** ensure the book stays honest:

1. **mdBook linkcheck preprocessor** — catches dead internal links and broken docs.rs references at build time. Build fails on dead link.
2. **`{{#include}}` referenced files must exist** — mdBook errors out if you reference a missing example file. The book breaks if someone deletes/renames an example without updating the chapter.
3. **Examples + doctests already in the existing CI gate** — `cargo check --examples` and `cargo test --doc -p heartbit-core`. So if a public API changes incompatibly, examples and doctests fail, signalling the book chapter needs a prose update.

### Tone & Voice

- **Tutorial-grade narrative** for early chapters (Introduction, Getting Started, Agents, Tools) — explanatory, paced, lots of context for readers new to agentic programming.
- **Reference-style with cross-links** for later chapters (Workflow Agents, Configuration, Eval, Production) — assumes the reader has the basics, trades hand-holding for density.
- **Standard "Rust Programming Language" pacing**: each new concept gets one well-explained example; subsequent uses cross-link instead of re-explaining.
- **Active voice, second person** ("You build an agent by...") not passive ("Agents are built by...").
- **No emoji, no marketing language.** Match the tone of the rewritten top-level README and the existing `crates/heartbit-core/README.md`.

### Maintenance

- Update `CONTRIBUTING.md` with one line: "If you change a public API in `heartbit-core`, update the relevant chapter in `book/src/`."
- The CI gate (linkcheck + `{{#include}}` resolution + existing example/doctest compile checks) catches the most common drift. Prose can still go stale; that's a known limitation of any narrative documentation.
- Maintenance ownership stays with the project owner for now. If contributor activity grows, the social contract above plus a docs-checking PR template is enough until it isn't.

## Sequencing & Implementation

~19 conceptual tasks, ordered for clean PR review:

1. **Scaffold `book/`**: `book.toml`, empty `src/SUMMARY.md`, placeholder `src/introduction.md`. Verify `mdbook build` succeeds.
2. **Add CI workflow** `.github/workflows/book.yml`. Verify it builds on PR.
3. **Add 3–4 missing examples** (`sequential_agent.rs`, `dag_agent.rs`, `from_toml.rs`, optionally one recipe-specific). Verify they compile via existing gate.
4. **Add 6–8 doctests** to key public types. Verify `cargo test --doc -p heartbit-core` passes.
5. **Chapter 1 — Introduction** (~600 words; no code).
6. **Chapter 2 — Getting Started** (~800 words; uses `hello_agent.rs`; includes API-key/env-var section).
7. **Chapter 3 — Agents** (~900 words; uses `simple_agent.rs`).
8. **Chapter 4 — Tools** (~900 words; uses `custom_tool.rs` + `mcp_agent.rs` partial).
9. **Chapter 5 — Memory** (~800 words; uses `memory.rs`).
10. **Chapter 6 — Guardrails** (~800 words; uses `guardrails.rs`).
11. **Chapter 7 — Workflow Agents** (~700 words; uses new `sequential_agent.rs` + `dag_agent.rs`).
12. **Chapter 8 — Orchestration** (~700 words; uses `multi_agent.rs`).
13. **Chapter 9 — Configuration** (~600 words; uses `from_toml.rs` + references existing `heartbit.toml`).
14. **Chapter 10 — Eval** (~500 words; uses `eval.rs`).
15. **Chapter 11 — Recipes** (~1,200 words; 6 cookbook entries reusing existing examples + 1 new recipe).
16. **Chapter 12 — Production** (~400 words; mostly cross-links to `docs/platform.md` + `crates/heartbit-cli/README.md`).
17. **Polish pass**: run `mdbook serve` locally, eyeball every page for navigation/flow, fix dead links flagged by linkcheck.
18. **One-line `CONTRIBUTING.md` update** — note about updating chapters when public APIs change.
19. **DNS + Pages config** — performed by the project owner outside the code work (~5 min).

Each task is one commit. Tasks 5–16 (the chapters) can each ship as a separate small commit; PR squashes are also fine.

**Total estimated effort: 5–7 days** of focused writing. Tasks 1–4 are ~1 day of plumbing; tasks 5–16 are the real work (~4–5 days); tasks 17–18 are ~half a day.

## Risks

- **Tone drift across chapters** if multiple subagents write them in parallel. Mitigation: one subagent per chapter; the spec specifies tone guidelines; the implementation plan instructs each subagent to skim adjacent chapters before writing.
- **Examples become stale** between when they're added (task 3) and when they're referenced from the book (tasks 6+). Mitigation: examples are added in the same task as the chapter that uses them, not as a separate up-front batch.
- **DNS propagation delay** could mean the published Pages site shows the GitHub default URL before `docs.heartbit.ai` resolves. Cosmetic; resolves itself within hours of the DNS change.
- **mdBook plugin compatibility** — `mdbook-mermaid` and `mdbook-linkcheck` both have `cargo install` steps. CI cold-starts will be slow (~3 min) but cached on subsequent runs. If either plugin breaks against a future mdBook version, the build halts; pin versions if it becomes flaky.
- **Linkcheck false positives** on docs.rs links if the crate hasn't been published yet. Mitigation: publishable cargo-publish-able state was verified in B3 Task 12; if needed, add `mdbook-linkcheck` config to ignore docs.rs/heartbit-core links until the first crates.io release.

## Out-of-Scope

These items are NOT addressed in this round:

- Platform/operations docs (lives in `docs/platform.md` + `crates/heartbit-cli/README.md`; linked from book's Production chapter).
- Comparison with LangChain / Mastra / Eliza.
- Translations.
- Versioned documentation.
- Replacing rustdoc with the book.
- Custom theming, branding, logos beyond the default mdBook theme + `CNAME`.
- SEO beyond mdBook's default sitemap.
- Heavy interactive widgets (live REPL, runnable browser examples, etc.) — defer.

## Exit Criteria

1. `cd book && mdbook build` succeeds locally.
2. CI workflow builds and (on `main`) deploys to `docs.heartbit.ai`.
3. The book's table of contents matches the locked TOC above.
4. Every code block in the book either uses `{{#include}}` to pull from a real `examples/*.rs` file or is a doctest cross-referenced from rustdoc.
5. `cargo check --examples` and `cargo test --doc -p heartbit-core` both pass.
6. mdBook linkcheck reports zero dead internal links.
7. `CONTRIBUTING.md` has the one-line maintenance note.
8. DNS for `docs.heartbit.ai` is set up and resolves.

## Public API Additions

None. The book has no public API surface — it's documentation. Doctests added in source code are internal `///` examples; they don't change the public API of `heartbit-core`.
