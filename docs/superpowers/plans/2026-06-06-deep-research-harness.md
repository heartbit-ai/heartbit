# Deep-Research Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `deep_research` workflow recipe (plan → parallel search/read with per-angle web tools → verify → synthesize cited report) + a deterministic `/research` TUI command.

**Architecture:** The recipe lives in a new focused module `crates/heartbit-core/src/agent/deep_research.rs`, registered in `default_registry()` beside `parallel_review`. Angle agents get their OWN `WebSearchTool`/`WebFetchTool` via `AgentCall::tools` (the `RunWorkflowTool` ctx stays tool-less). The TUI command is reducer-only: it builds an imperative single-purpose order through the standard send path.

**Tech Stack:** existing flow combinators (`agent`, `parallel`, `thunk`), `WorkflowRecipe`/`WorkflowRegistry`, websearch/webfetch builtins. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-06-deep-research-harness-design.md`

---

## Verified ground truth (do not re-derive)

- `agent(&ctx, prompt) -> AgentCall<NoSchema>`; chainable `.tools(Vec<Arc<dyn Tool>>)` (flow/agent.rs:128), `.label(String)`; `.run().await -> Result<Option<String>, Error>` (flow/agent.rs:287).
- `parallel(&ctx, thunks).await -> Vec<Option<Option<String>>>` for thunks returning `Option<String>` — parallel_review flattens each slot (`slot.flatten()`); a thunk's error becomes `None` (fail-soft).
- `thunk(closure)` helper is imported in workflow_tool.rs (`use super::flow::{WorkflowCtx, agent, parallel, thunk};`).
- `WorkflowRecipe { name, description, args_schema, run: Arc<dyn Fn(WorkflowCtx, Value) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send>> ... }` — see `parallel_review()` at workflow_tool.rs:181 for the exact shape to mirror.
- `default_registry()` at workflow_tool.rs:259: `WorkflowRegistry::new().register(recipes::parallel_review())`.
- `WebSearchTool::try_new() -> Result<Self, Error>` (websearch.rs:51); `WebFetchTool::try_new() -> Result<Self, Error>` (webfetch.rs:44). Both `Arc<dyn Tool>`-able.
- workflow_tool tests use a local `AlwaysText` provider + `fn provider() -> Arc<BoxedProvider>`; the recipe test needs a NEW content-routed capturing mock (defined in Task 2) because the parallel stage makes call ORDER nondeterministic — route responses on prompt substrings, never on call index.
- TUI: `handle_slash` dispatch + no-key guard precedent (the `"learn"` arm in app.rs); send-path precedent (`Msg::AnalyzeReady` reducer arm: `Cell::User(display)`, `running=true`, `follow=true`, `seed_idle_squad()`, `Effect::SendInput(task)`). Test helpers `key()`, `typed()` (+Enter), `keyed()`.
- `.gitignore` already has a heartbit-tui artifacts block (`heartbit-session-*.md` etc.).
- `agent/mod.rs` declares the agent submodules (`mod deep_research;` goes there); `default_registry` must reference `crate::agent::deep_research::recipe()`.

## File structure

- **Create** `crates/heartbit-core/src/agent/deep_research.rs` — angle parser (pure) + the recipe + tests.
- **Modify** `crates/heartbit-core/src/agent/mod.rs` — `mod deep_research;`.
- **Modify** `crates/heartbit-core/src/agent/workflow_tool.rs` — register the recipe in `default_registry()` + registry test update.
- **Modify** `crates/heartbit-tui/src/app.rs` — `/research` command (slug helper, SLASH_COMMANDS, handle_slash arm, tests).
- **Modify** `.gitignore` — `research-*.md`.

---

### Task 1: angle parser (pure, TDD)

**Files:** Create `crates/heartbit-core/src/agent/deep_research.rs`; modify `crates/heartbit-core/src/agent/mod.rs` (add `mod deep_research;` alongside the other private agent modules).

- [x] **Step 1: failing tests.** Create the file:

```rust
//! The `deep_research` workflow recipe: plan → parallel search/read (each
//! angle agent carries its own websearch/webfetch tools) → cross-verify →
//! synthesize a cited report. Born from a live failure (session 6a245538):
//! asked to "deep research", the agent had no harness to route to, its
//! scraped searches died silently, and it fabricated URLs.

use std::sync::Arc;

use serde_json::{Value, json};

use super::flow::{agent, parallel, thunk};
use super::workflow_tool::WorkflowRecipe;
use crate::error::Error;

/// Clamp bounds for the `angles` argument.
const MIN_ANGLES: usize = 2;
const MAX_ANGLES: usize = 6;
const DEFAULT_ANGLES: usize = 4;

/// Parse the planning agent's angle list: accepts `1. foo` / `1) foo` /
/// `- foo` / `* foo` lines, trims, drops empties, caps at `max`. Returns the
/// deterministic fallback (the question + a state-of-the-art variant) when
/// fewer than [`MIN_ANGLES`] parse — the plan stage can never fail the run.
fn parse_angles(text: &str, max: usize, question: &str) -> Vec<String> {
    let mut angles: Vec<String> = text
        .lines()
        .map(|l| {
            l.trim()
                .trim_start_matches(|c: char| c.is_ascii_digit())
                .trim_start_matches(['.', ')', '-', '*'])
                .trim()
                .to_string()
        })
        .filter(|l| !l.is_empty())
        .take(max)
        .collect();
    if angles.len() < MIN_ANGLES {
        angles = vec![
            question.to_string(),
            format!("state of the art: {question}"),
        ];
    }
    angles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_angles_accepts_numbered_and_bulleted() {
        let text = "1. definition and use cases\n2) core algorithms\n- existing implementations\n* pitfalls\n\n";
        let a = parse_angles(text, 6, "q");
        assert_eq!(
            a,
            vec![
                "definition and use cases",
                "core algorithms",
                "existing implementations",
                "pitfalls"
            ]
        );
    }

    #[test]
    fn parse_angles_caps_at_max() {
        let text = "1. a\n2. b\n3. c\n4. d\n5. e";
        assert_eq!(parse_angles(text, 3, "q").len(), 3);
    }

    #[test]
    fn parse_angles_falls_back_on_garbage() {
        let a = parse_angles("I will now think about this.", 4, "plate solving");
        // One prose line parses as one "angle" — below MIN_ANGLES → fallback.
        assert_eq!(a.len(), 2);
        assert_eq!(a[0], "plate solving");
        assert!(a[1].contains("state of the art"));
        let b = parse_angles("", 4, "q");
        assert_eq!(b.len(), 2);
    }
}
```

In `crates/heartbit-core/src/agent/mod.rs`, add `mod deep_research;` next to the other `mod` lines (alphabetical placement near `mod doom_loop;`).

- [x] **Step 2:** Run `cargo test -p heartbit-core deep_research::tests` — the file as written above already contains the impl, so write tests FIRST in your working copy (paste the tests module + a `todo!()`-free stub? NO): to honour TDD, create the file with ONLY the module doc, the consts, an EMPTY `parse_angles` body returning `Vec::new()`, and the tests — observe the assertion failures — then fill the real body shown above. Expected red: `parse_angles_accepts_numbered_and_bulleted` fails with left `[]`.

- [x] **Step 3:** Replace the stub body with the real implementation (shown in Step 1). Run again: 3 PASS. Note: the file has unused imports (`Arc`, `json`, `agent`…) until Task 2 — if clippy `-D warnings` complains at this point, keep ONLY the used imports now and add the rest in Task 2 (do not blanket-allow).

- [x] **Step 4: commit**

```bash
git add crates/heartbit-core/src/agent/deep_research.rs crates/heartbit-core/src/agent/mod.rs
git commit -m "feat(core): deep_research angle parser (tolerant, fallback-guaranteed)"
```

---

### Task 2: the recipe + registry entry

**Files:** Modify `crates/heartbit-core/src/agent/deep_research.rs`, `crates/heartbit-core/src/agent/workflow_tool.rs`.

- [x] **Step 1: failing tests.** Append to deep_research.rs tests (and add the test-support mock at the top of the tests module):

```rust
    use crate::BoxedProvider;
    use crate::agent::flow::WorkflowCtx;
    use crate::llm::LlmProvider;
    use crate::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };
    use std::sync::Mutex;

    /// Content-routed mock: the parallel stage makes call ORDER
    /// nondeterministic, so responses are selected by prompt substring,
    /// never by call index. Captures every request for the tools-wiring
    /// assertions.
    struct RoutedProvider {
        captured: Mutex<Vec<CompletionRequest>>,
    }

    impl RoutedProvider {
        fn new() -> Self {
            Self {
                captured: Mutex::new(Vec::new()),
            }
        }
        fn text(t: &str) -> CompletionResponse {
            CompletionResponse {
                content: vec![ContentBlock::Text { text: t.into() }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            }
        }
    }

    impl LlmProvider for RoutedProvider {
        async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
            let prompt = request
                .messages
                .iter()
                .flat_map(|m| m.content.iter())
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<String>();
            self.captured.lock().unwrap().push(request);
            let reply = if prompt.contains("Decompose the question") {
                "1. angle one\n2. angle two"
            } else if prompt.contains("FINDINGS:") {
                "FINDINGS:\n- fact X [https://ex.org/a]\nSOURCES:\n- https://ex.org/a"
            } else if prompt.contains("cross-check") {
                "CONFIRMED: fact X (2 sources)"
            } else if prompt.contains("final cited report") {
                "# Report\n\nfact X.\n\n## Sources\n- https://ex.org/a"
            } else {
                "unexpected prompt"
            };
            Ok(Self::text(reply))
        }
    }

    fn run_recipe(provider: Arc<RoutedProvider>, args: Value) -> Result<String, Error> {
        let ctx = WorkflowCtx::builder(Arc::new(BoxedProvider::new_from_arc(provider)))
            .build()
            .unwrap();
        let r = recipe();
        futures::executor::block_on((r.run)(ctx, args))
    }

    #[test]
    fn recipe_happy_path_produces_cited_report() {
        let provider = Arc::new(RoutedProvider::new());
        let report = run_recipe(
            provider.clone(),
            json!({"question": "how does plate solving work", "angles": 2}),
        )
        .unwrap();
        assert!(report.contains("## Sources"), "cited report: {report}");
        assert!(report.contains("ex.org"), "{report}");
    }

    #[test]
    fn angle_agents_carry_web_tools_and_other_stages_do_not() {
        let provider = Arc::new(RoutedProvider::new());
        let _ = run_recipe(provider.clone(), json!({"question": "q", "angles": 2})).unwrap();
        let reqs = provider.captured.lock().unwrap();
        assert!(reqs.len() >= 4, "plan + 2 angles + verify + synthesize");
        for r in reqs.iter() {
            let prompt: String = r
                .messages
                .iter()
                .flat_map(|m| m.content.iter())
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            let names: Vec<&str> = r.tools.iter().map(|t| t.name.as_str()).collect();
            if prompt.contains("FINDINGS:") {
                assert!(
                    names.contains(&"websearch") && names.contains(&"webfetch"),
                    "angle agents must carry web tools, got {names:?}"
                );
            } else {
                assert!(
                    !names.contains(&"websearch"),
                    "non-angle stages must stay tool-less, got {names:?}"
                );
            }
        }
    }

    #[test]
    fn recipe_requires_a_question() {
        let provider = Arc::new(RoutedProvider::new());
        let err = run_recipe(provider, json!({})).unwrap_err();
        assert!(err.to_string().contains("question"));
    }

    #[test]
    fn registry_includes_deep_research() {
        let reg = crate::agent::workflow_tool::default_registry();
        assert!(reg.get("deep_research").is_some());
    }
```

NOTE on helpers used above: check `BoxedProvider`'s constructor surface — if `new_from_arc` does not exist, wrap the mock the way workflow_tool tests do (`BoxedProvider::new(...)` consumes the provider by value; then capture must go through an `Arc<Mutex<…>>` HELD OUTSIDE and cloned into the mock: give `RoutedProvider` a `captured: Arc<Mutex<Vec<CompletionRequest>>>` field, keep a clone in the test before moving the provider into `BoxedProvider::new`). Likewise if `futures::executor::block_on` is unavailable, use `#[tokio::test]` + `.await` (the flow is tokio-based — PREFER `#[tokio::test(flavor = "multi_thread")]` and make `run_recipe` async). Adapt mechanically; the assertions are the contract.

- [x] **Step 2:** Run — red (`recipe` undefined, registry missing entry).

- [x] **Step 3: implement the recipe** in deep_research.rs:

```rust
/// Build the `deep_research` recipe. Stage agents are talk-only EXCEPT the
/// angle agents, which carry their own websearch/webfetch instances via
/// `AgentCall::tools` — the `run_workflow` ctx itself stays tool-less.
pub(crate) fn recipe() -> WorkflowRecipe {
    WorkflowRecipe {
        name: "deep_research".into(),
        description: "Research-first deep dive: decompose the question into \
                      angles, search and read sources per angle (with real web \
                      tools), cross-verify claims, and synthesize a cited \
                      report. Use whenever the user asks for deep research / \
                      état de l'art / a sourced investigation."
            .into(),
        args_schema: json!({
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The research question."},
                "angles": {"type": "integer", "description": "Number of research angles (2-6, default 4)."}
            },
            "required": ["question"]
        }),
        run: Arc::new(|ctx, args| {
            Box::pin(async move {
                let question = args
                    .get("question")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|q| !q.is_empty())
                    .ok_or_else(|| Error::Agent("deep_research: 'question' is required".into()))?
                    .to_string();
                let n_angles = args
                    .get("angles")
                    .and_then(|v| v.as_u64())
                    .map(|n| (n as usize).clamp(MIN_ANGLES, MAX_ANGLES))
                    .unwrap_or(DEFAULT_ANGLES);

                // Stage 1 — plan (talk-only; parse is fallback-guaranteed).
                let plan = agent(
                    &ctx,
                    format!(
                        "Decompose the question below into {n_angles} complementary \
                         RESEARCH ANGLES (e.g. definition/state of the art, \
                         algorithms/methods, existing implementations, \
                         pitfalls/limits). Output ONLY the angles, one per line, \
                         numbered.\n\nQuestion: {question}"
                    ),
                )
                .label("research:plan".into())
                .run()
                .await?
                .unwrap_or_default();
                let angles = parse_angles(&plan, n_angles, &question);

                // Stage 2 — search+read, one tooled agent per angle (parallel,
                // fail-soft: a dead angle becomes degraded coverage, not a crash).
                let thunks: Vec<_> = angles
                    .iter()
                    .cloned()
                    .map(|angle| {
                        let ctx = ctx.clone();
                        let question = question.clone();
                        thunk(move || async move {
                            let mut tools: Vec<Arc<dyn crate::tool::Tool>> = Vec::new();
                            if let Ok(t) = crate::tool::builtins::WebSearchTool::try_new() {
                                tools.push(Arc::new(t));
                            }
                            if let Ok(t) = crate::tool::builtins::WebFetchTool::try_new() {
                                tools.push(Arc::new(t));
                            }
                            agent(
                                &ctx,
                                format!(
                                    "Research this angle of \"{question}\":\n  {angle}\n\n\
                                     1. Run 1-2 websearch queries for the angle.\n\
                                     2. Pick the 1-2 most authoritative results and webfetch them.\n\
                                     3. Extract concrete findings, each with its [URL] citation.\n\n\
                                     Output EXACTLY two sections:\nFINDINGS:\n- claim [URL]\n…\nSOURCES:\n- URL\n…\n\n\
                                     If search or fetch FAILS (blocked provider, 404), say so under \
                                     FINDINGS — NEVER invent URLs or facts."
                                ),
                            )
                            .tools(tools)
                            .label(format!("research:angle:{angle}"))
                            .run()
                            .await
                        })
                    })
                    .collect();
                let results = parallel(&ctx, thunks).await;
                let notes = angles
                    .iter()
                    .zip(results)
                    .map(|(angle, slot)| {
                        let body = slot
                            .flatten()
                            .unwrap_or_else(|| "(angle produced no findings)".to_string());
                        format!("### Angle: {angle}\n{body}")
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
                if !notes.contains('[') && !notes.contains("FINDINGS") {
                    return Err(Error::Agent(
                        "deep_research: every research angle failed — check the \
                         search provider (see the startup 'search:' line)"
                            .into(),
                    ));
                }

                // Stage 3 — verify (talk-only).
                let verification = agent(
                    &ctx,
                    format!(
                        "cross-check the research notes below. Classify each claim: \
                         CONFIRMED (multiple sources), SINGLE-SOURCE, or CONTRADICTED \
                         (cite the conflicting sources). List notable gaps.\n\n{notes}"
                    ),
                )
                .label("research:verify".into())
                .run()
                .await?
                .unwrap_or_default();

                // Stage 4 — synthesize (talk-only).
                let report = agent(
                    &ctx,
                    format!(
                        "Write the final cited report in Markdown for the question \
                         \"{question}\".\nSections: # <title> · Summary · Findings \
                         (each claim tagged CONFIRMED/SINGLE-SOURCE/CONTRADICTED with \
                         its [URL] citations) · Contradictions & open questions · \
                         ## Sources (deduplicated URL list).\nUse ONLY the material \
                         below — no invented facts or URLs.\n\n## Research notes\n\
                         {notes}\n\n## Verification\n{verification}"
                    ),
                )
                .label("research:synthesize".into())
                .run()
                .await?
                .unwrap_or_default();
                if report.trim().is_empty() {
                    return Err(Error::Agent("deep_research: synthesis produced no report".into()));
                }
                Ok(report)
            })
        }),
    }
}
```

Check visibility while wiring: `WorkflowRecipe`'s fields and `workflow_tool`'s module path (`pub(crate)` as needed — `default_registry` lives in `workflow_tool.rs` and must call `super::deep_research::recipe()` or `crate::agent::deep_research::recipe()`); `WebFetchTool` must be exported from `tool::builtins` (it is: check `pub use` list; if absent, add it beside `WebSearchTool`).

- [x] **Step 4:** Registry: in `workflow_tool.rs`, change `default_registry()`:

```rust
pub fn default_registry() -> WorkflowRegistry {
    WorkflowRegistry::new()
        .register(recipes::parallel_review())
        .register(crate::agent::deep_research::recipe())
}
```

…and extend the existing `registry_get_and_meta` test: `assert!(reg.get("deep_research").is_some());`.

- [x] **Step 5:** `cargo test -p heartbit-core deep_research` + `cargo test -p heartbit-core workflow_tool` — all PASS. Then `cargo clippy -p heartbit-core --all-targets -- -D warnings` + `cargo fmt --all`.

- [x] **Step 6: commit**

```bash
git add crates/heartbit-core/src/agent/deep_research.rs crates/heartbit-core/src/agent/workflow_tool.rs crates/heartbit-core/src/tool/builtins/mod.rs
git commit -m "feat(core): deep_research workflow recipe — plan, tooled angles, verify, cited synthesis"
```

---

### Task 3: `/research` TUI command

**Files:** Modify `crates/heartbit-tui/src/app.rs`, `.gitignore`.

- [x] **Step 1: failing tests** (app.rs tests module):

```rust
    #[test]
    fn research_slug_is_safe_and_bounded() {
        assert_eq!(research_slug("How does Plate Solving work?"), "how-does-plate-solving-work");
        assert_eq!(research_slug("éàç!!"), "research");
        assert!(research_slug(&"x".repeat(200)).len() <= 40);
    }

    #[test]
    fn slash_research_builds_the_imperative_task() {
        let mut app = keyed();
        typed(&mut app, "/research plate solving algorithms");
        app.update(key(KeyCode::Enter));
        assert!(matches!(app.history.last(), Some(Cell::User(t)) if t.contains("researching")));
        assert!(app.running);
        let task = app
            .effects
            .iter()
            .find_map(|e| match e {
                Effect::SendInput(t) => Some(t.clone()),
                _ => None,
            })
            .expect("task sent");
        assert!(task.contains("run_workflow"), "{task}");
        assert!(task.contains("deep_research"), "{task}");
        assert!(task.contains("research-plate-solving-algorithms.md"), "{task}");
        assert!(task.to_lowercase().contains("do not improvise"), "{task}");
    }

    #[test]
    fn slash_research_empty_arg_is_usage_no_key_is_modal() {
        let mut app = keyed();
        typed(&mut app, "/research");
        app.update(key(KeyCode::Enter));
        assert!(matches!(app.history.last(), Some(Cell::Notice(n)) if n.contains("usage")));
        assert!(!app.running);
        let mut app = App::new("m");
        typed(&mut app, "/research topic");
        app.update(key(KeyCode::Enter));
        assert!(matches!(app.modal, Some(Modal::KeyEntry(_))));
        assert!(!app.running);
    }
```

- [x] **Step 2:** red (`research_slug` undefined, no arm).

- [x] **Step 3: implement** in app.rs:

Slug helper (free function near the top-level helpers, with doc):

```rust
/// Workspace-safe slug for `/research` artifacts: lowercase alphanumerics
/// joined by single dashes, capped at 40 chars; degenerate input → "research".
fn research_slug(question: &str) -> String {
    let mut slug = String::new();
    let mut dash = false;
    for c in question.chars().take(80).flat_map(char::to_lowercase) {
        if c.is_ascii_alphanumeric() {
            slug.push(c);
            dash = false;
        } else if !dash && !slug.is_empty() {
            slug.push('-');
            dash = true;
        }
        if slug.len() >= 40 {
            break;
        }
    }
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() { "research".into() } else { slug }
}
```

`SLASH_COMMANDS` (after `/learn`): `("/research", "deep research — fan-out, verify, cited report"),`

`handle_slash` arm (before `other =>`):

```rust
            "research" => {
                if arg.is_empty() {
                    self.history.push(Cell::Notice(
                        "usage: /research <question> — fan-out research, cross-verify, cited report"
                            .into(),
                    ));
                    return;
                }
                if self.api_key.is_none() && !self.has_fallback_provider {
                    self.open_key_modal();
                    return;
                }
                let slug = research_slug(&arg);
                let task = format!(
                    "Call the run_workflow tool now with name=\"deep_research\" and \
                     args={{\"question\": {q}}}. Do NOT search, browse, or implement \
                     anything yourself before the workflow returns. When it returns, \
                     write the report verbatim to research-{slug}.md (workspace-relative \
                     path) with the write tool, then give a 5-10 line summary of the key \
                     findings and sources. If the workflow returns an error, report it — \
                     do not improvise your own research.",
                    q = serde_json::to_string(&arg).unwrap_or_else(|_| format!("\"{arg}\"")),
                );
                self.history
                    .push(Cell::User(format!("researching: {arg}")));
                self.running = true;
                self.follow = true;
                self.seed_idle_squad();
                self.effects.push(Effect::SendInput(task));
            }
```

`.gitignore`: append `research-*.md` to the heartbit-tui artifacts block.

- [x] **Step 4:** `cargo test -p heartbit-tui` all PASS; fmt + clippy clean.

- [x] **Step 5: commit**

```bash
git add crates/heartbit-tui/src/app.rs .gitignore
git commit -m "feat(tui): /research — deterministic deep_research trigger with cited-report contract"
```

---

### Task 4: workspace gate

- [x] `cargo fmt --all -- --check && cargo clippy --workspace --exclude mini-crm --all-targets -- -D warnings && cargo test --workspace --exclude mini-crm` — all green. Commit only if fixes were needed.

---

### Task 5: live validation (pty)

Per the project bar (settled frame, trace-grounded assertions). Requires the Exa key reaching the process: write it into `~/.config/heartbit/tui.toml` as `exa_api_key = "…"` (the seam shipped today) from `.env`, or export it into the pty env.

- [x] **Step 1:** `cargo build -p heartbit-tui`. Fresh temp cwd, real session, `/mode yolo`, then `/research what is plate solving in astrophotography` and wait (budget ≥ 300s — 4 angles × search+fetch + synthesis).
- [x] **Step 2:** Assert: `research-what-is-plate-solving-*.md` exists in the cwd, non-empty, contains `## Sources` and at least one `http` URL; the transcript's settled frame contains a summary (and NOT a fabricated-looking 404 wall); the trace contains a `run_workflow` tool_call with `is_error:false` and websearch/webfetch tool_calls from the recipe run.
- [x] **Step 3:** Degraded path: one session WITHOUT any search key (unset env, no tui.toml key) → `/research x` → the report or the error message must be HONEST (mentions blocked/failed search; no invented URLs). The startup line must show `search: ddg-only (no search API key)`.
- [x] **Step 4:** Restore the user's tui.toml to their preference; report results.

---

## Self-review

1. **Spec coverage:** recipe stages + prompts ✓ (T2, verbatim anti-fabrication rule) · per-angle tools via AgentCall::tools, ctx stays tool-less ✓ (T2 + wiring test) · angles clamp 2..=6 default 4 ✓ · plan fallback ✓ (T1) · all-angles-dead → Err ✓ (T2 guard + spec wording) · registry entry ✓ · /research with usage + no-key guard + imperative order naming the slug file ✓ (T3) · slug helper ✓ · .gitignore ✓ · observability = free (tool call) ✓ · live validation incl. degraded path ✓ (T5).
2. **Placeholders:** none — full code in every step; the two API uncertainties (BoxedProvider arc-wrapping, block_on vs tokio::test) are named with concrete adaptation instructions, not deferred.
3. **Type consistency:** `recipe()` (T2) referenced by registry (T2 Step 4) and test; `research_slug` (T3) used in arm + tests; `parse_angles(text, max, question)` signature consistent T1/T2.
4. **Known judgment call:** the all-angles-dead detector (`!notes.contains('[') && !notes.contains("FINDINGS")`) is heuristic — angle agents that honestly report failure still emit a FINDINGS section, so a fully-dead run only trips when every slot flattened to the "(angle produced no findings)" placeholder or empty prose. If the implementer finds a crisper signal (e.g. count slots that returned None), prefer counting None slots ≥ angles.len() — equivalent intent, cleaner.
