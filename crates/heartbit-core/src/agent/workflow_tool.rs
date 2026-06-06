//! Named workflow registry + the `run_workflow` tool (option C, tier 3).
//!
//! The `flow/` combinators are developer-authored Rust — the LLM cannot write a
//! workflow at runtime. So the agent reaches workflows the Claude-Code way:
//! a developer pre-registers named **recipes**, and the agent picks one by name
//! plus args via the `run_workflow` tool. A recipe drives the `flow/`
//! combinators (agent, parallel, pipeline) and returns a text result — the
//! structured, repeatable fan-out that plain `delegate_task` lacks (multi-stage
//! pipeline, shared budget, journal/resume).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::{Value, json};

use super::flow::{WorkflowCtx, agent, parallel, thunk};
use crate::error::Error;
use crate::llm::BoxedProvider;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

/// The async body of a recipe: given a built [`WorkflowCtx`] and the
/// LLM-supplied `args`, drive `flow/` combinators and return a text result.
pub type RecipeRun = Arc<
    dyn Fn(WorkflowCtx, Value) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send>>
        + Send
        + Sync,
>;

/// A named, agent-invocable workflow recipe.
#[derive(Clone)]
pub struct WorkflowRecipe {
    /// Stable identifier the agent passes as `recipe`.
    pub name: String,
    /// One-line description shown to the agent (how/when to use it).
    pub description: String,
    /// JSON Schema for the recipe's `args` (advisory — shown to the agent).
    pub args_schema: Value,
    /// The recipe body.
    pub run: RecipeRun,
}

/// Registry of recipes the [`RunWorkflowTool`] exposes.
#[derive(Clone, Default)]
pub struct WorkflowRegistry {
    recipes: Vec<WorkflowRecipe>,
}

impl WorkflowRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a recipe (builder-style).
    pub fn register(mut self, recipe: WorkflowRecipe) -> Self {
        self.recipes.push(recipe);
        self
    }

    /// Look up a recipe by name.
    pub fn get(&self, name: &str) -> Option<&WorkflowRecipe> {
        self.recipes.iter().find(|r| r.name == name)
    }

    /// True when no recipes are registered (→ the tool is not registered).
    pub fn is_empty(&self) -> bool {
        self.recipes.is_empty()
    }

    /// `(name, description)` pairs to advertise in the entry-agent prompt.
    pub fn meta(&self) -> Vec<(String, String)> {
        self.recipes
            .iter()
            .map(|r| (r.name.clone(), r.description.clone()))
            .collect()
    }
}

/// The `run_workflow` tool: the agent picks a recipe by name + args.
pub struct RunWorkflowTool {
    registry: WorkflowRegistry,
    provider: Arc<BoxedProvider>,
    agent_events: Option<Arc<crate::agent::events::OnEvent>>,
    provider_factory: Option<Arc<super::flow::ProviderFactory>>,
    journal_dir: Option<std::path::PathBuf>,
    workspace: Option<std::path::PathBuf>,
}

impl RunWorkflowTool {
    /// Build the tool over a registry and the shared provider used to run the
    /// recipe's `flow/` agents.
    pub fn new(registry: WorkflowRegistry, provider: Arc<BoxedProvider>) -> Self {
        Self {
            registry,
            provider,
            agent_events: None,
            provider_factory: None,
            journal_dir: None,
            workspace: None,
        }
    }

    /// Forward every recipe-internal agent's event stream (tool calls, LLM
    /// responses, run lifecycle) to `sink` — e.g. the TUI trace. Without this,
    /// a recipe run is a single opaque tool call to any observer.
    pub fn with_agent_events(mut self, sink: Arc<crate::agent::events::OnEvent>) -> Self {
        self.agent_events = Some(sink);
        self
    }

    /// Install the per-call model resolver threaded into every recipe ctx —
    /// lets recipes run stages on role-named models (`.model("fast")`).
    pub fn with_provider_factory(mut self, factory: Arc<super::flow::ProviderFactory>) -> Self {
        self.provider_factory = Some(factory);
        self
    }

    /// Workspace (git repo root) threaded into every recipe ctx — required
    /// for recipes whose agents ask for `Isolation::Worktree`.
    pub fn with_workspace(mut self, root: std::path::PathBuf) -> Self {
        self.workspace = Some(root);
        self
    }

    /// Enable resume: each run journals its agent outputs under `dir`, keyed
    /// by recipe + canonicalized args. Re-invoking the SAME call replays the
    /// completed agents at zero cost and continues the rest — an interrupted
    /// workflow picks up where it left off (scope the dir per session to avoid
    /// stale cross-session replays).
    pub fn with_journal_dir(mut self, dir: std::path::PathBuf) -> Self {
        self.journal_dir = Some(dir);
        self
    }
}

/// `wf-{recipe}-{sha256(canonical args)[..12]}.jsonl` — content-addressed so a
/// re-ask with identical inputs resumes, while ANY args change runs fresh.
fn journal_file_name(recipe: &str, args: &Value) -> String {
    use sha2::{Digest, Sha256};
    let canonical = super::flow::journal::canonical_json(args);
    let mut h = Sha256::new();
    h.update(canonical.to_string().as_bytes());
    let digest = h.finalize();
    let hex: String = digest.iter().take(6).map(|b| format!("{b:02x}")).collect();
    format!("wf-{recipe}-{hex}.jsonl")
}

impl Tool for RunWorkflowTool {
    fn definition(&self) -> ToolDefinition {
        let names: Vec<String> = self
            .registry
            .recipes
            .iter()
            .map(|r| r.name.clone())
            .collect();
        let list = self
            .registry
            .recipes
            .iter()
            .map(|r| format!("- {}: {}", r.name, r.description))
            .collect::<Vec<_>>()
            .join("\n");
        ToolDefinition {
            name: "run_workflow".into(),
            description: format!(
                "Launch a named multi-step workflow recipe for structured, repeatable fan-out \
                 (parallel/staged sub-agents with a shared budget). Pick one by name and pass \
                 its args. Recipes:\n{list}"
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "recipe": {
                        "type": "string",
                        "enum": names,
                        "description": "Which recipe to run."
                    },
                    "args": {
                        "type": "object",
                        "description": "Recipe-specific arguments (see the recipe description)."
                    },
                    "budget": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional hard token budget for the whole workflow \
                                        (cost-weighted token-equivalents). Use when the user \
                                        asks to cap spend, e.g. 'use 10k tokens' => 10000."
                    }
                },
                "required": ["recipe"]
            }),
        }
    }

    fn execute(
        &self,
        _ctx: &crate::ExecutionContext,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        let recipe_name = input
            .get("recipe")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let args = input.get("args").cloned().unwrap_or_else(|| json!({}));

        let budget = input.get("budget").and_then(|v| v.as_u64());
        let recipe = self.registry.get(&recipe_name).cloned();
        let provider = self.provider.clone();
        let agent_events = self.agent_events.clone();
        let provider_factory = self.provider_factory.clone();
        let journal_dir = self.journal_dir.clone();
        let workspace = self.workspace.clone();
        Box::pin(async move {
            let Some(recipe) = recipe else {
                return Ok(ToolOutput::error(format!(
                    "unknown workflow recipe '{recipe_name}'"
                )));
            };
            let mut builder = WorkflowCtx::builder(provider);
            if let Some(sink) = agent_events {
                builder = builder.on_agent_event(sink);
            }
            if let Some(n) = budget {
                builder = builder.budget(n);
            }
            if let Some(factory) = provider_factory {
                builder = builder.provider_factory(factory);
            }
            if let Some(root) = workspace {
                builder = builder.workspace(root);
            }
            if let Some(dir) = journal_dir {
                if let Err(e) = std::fs::create_dir_all(&dir) {
                    return Ok(ToolOutput::error(format!("workflow journal dir: {e}")));
                }
                let path = dir.join(journal_file_name(&recipe_name, &args));
                builder = match builder.journal(&path, super::flow::journal::ResumeMode::Resume) {
                    Ok(b) => b,
                    Err(e) => {
                        return Ok(ToolOutput::error(format!("workflow journal: {e}")));
                    }
                };
            }
            let ctx = match builder.build() {
                Ok(c) => c,
                Err(e) => return Ok(ToolOutput::error(format!("workflow setup failed: {e}"))),
            };
            let result = (recipe.run)(ctx.clone(), args).await;
            // A run-level breach (budget / agent backstop) must surface even
            // when the recipe collapsed the failed slots into degraded text.
            if let Some(breach) = ctx.control_breach() {
                return Ok(ToolOutput::error(format!(
                    "workflow '{recipe_name}' halted by a run-level limit: {breach:?}"
                )));
            }
            match result {
                Ok(text) => Ok(ToolOutput::success(text)),
                Err(e) => Ok(ToolOutput::error(format!(
                    "workflow '{recipe_name}' failed: {e}"
                ))),
            }
        })
    }
}

/// Built-in recipes.
pub mod recipes {
    use super::*;

    /// Default lenses applied by `parallel_review` when the caller gives none.
    const DEFAULT_LENSES: &[&str] = &["correctness", "security", "clarity"];

    /// Review a target (a file path, a diff, or a described change) from several
    /// INDEPENDENT lenses in parallel, then concatenate the findings. Exercises
    /// the `flow::parallel` combinator under the shared budget.
    ///
    /// Args: `{ "target": string (required), "lenses"?: [string] }`.
    pub fn parallel_review() -> WorkflowRecipe {
        WorkflowRecipe {
            name: "parallel_review".into(),
            description: "Review a target (file, diff, or described change) from multiple \
                          independent lenses (correctness/security/clarity) in parallel, then \
                          synthesize the findings."
                .into(),
            args_schema: json!({
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "What to review (path, diff, or description)."},
                    "lenses": {"type": "array", "items": {"type": "string"}, "description": "Review perspectives (default: correctness, security, clarity)."}
                },
                "required": ["target"]
            }),
            run: Arc::new(|ctx, args| {
                Box::pin(async move {
                    let target = args
                        .get("target")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    if target.trim().is_empty() {
                        return Err(Error::Agent("parallel_review: 'target' is required".into()));
                    }
                    let lenses: Vec<String> = args
                        .get("lenses")
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|x| x.as_str().map(String::from))
                                .collect::<Vec<_>>()
                        })
                        .filter(|v| !v.is_empty())
                        .unwrap_or_else(|| DEFAULT_LENSES.iter().map(|s| s.to_string()).collect());

                    let thunks: Vec<_> = lenses
                        .iter()
                        .cloned()
                        .map(|lens| {
                            let ctx = ctx.clone();
                            let target = target.clone();
                            thunk(move || async move {
                                agent(
                                    &ctx,
                                    format!(
                                        "Review the following from the **{lens}** perspective \
                                         only. Be concise and concrete — list specific issues or \
                                         confirm it is sound.\n\n{target}"
                                    ),
                                )
                                .label(format!("review:{lens}"))
                                .run()
                                .await
                            })
                        })
                        .collect();

                    let results = parallel(&ctx, thunks).await;

                    let findings = lenses
                        .iter()
                        .zip(results)
                        .map(|(lens, slot)| {
                            let body = slot.flatten().unwrap_or_else(|| "(no output)".to_string());
                            format!("### {lens}\n{body}")
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n");

                    Ok(format!("# Parallel review of: {target}\n\n{findings}"))
                })
            }),
        }
    }
}

/// A default registry for the TUI entry agent: the built-in recipes.
pub fn default_registry() -> WorkflowRegistry {
    WorkflowRegistry::new()
        .register(recipes::parallel_review())
        .register(crate::agent::deep_research::recipe())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExecutionContext;
    use crate::llm::LlmProvider;
    use crate::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };

    #[test]
    fn registry_get_and_meta() {
        let reg = default_registry();
        assert!(!reg.is_empty());
        assert!(reg.get("parallel_review").is_some());
        assert!(reg.get("deep_research").is_some());
        assert!(reg.get("nope").is_none());
        let meta = reg.meta();
        assert!(meta.iter().any(|(n, _)| n == "parallel_review"));
    }

    /// Always answers with a fixed text — order-independent for concurrent calls.
    struct AlwaysText(String);
    impl LlmProvider for AlwaysText {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: self.0.clone(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            })
        }
    }

    fn provider() -> Arc<BoxedProvider> {
        Arc::new(BoxedProvider::new(AlwaysText("LENS-OK".into())))
    }

    #[test]
    fn tool_definition_lists_recipe_names() {
        let tool = RunWorkflowTool::new(default_registry(), provider());
        let def = tool.definition();
        assert_eq!(def.name, "run_workflow");
        let enum_names = def.input_schema["properties"]["recipe"]["enum"]
            .as_array()
            .unwrap();
        assert!(enum_names.iter().any(|v| v == "parallel_review"));
    }

    #[tokio::test]
    async fn unknown_recipe_is_an_error_output() {
        let tool = RunWorkflowTool::new(default_registry(), provider());
        let out = tool
            .execute(
                &ExecutionContext::default(),
                json!({"recipe": "does_not_exist"}),
            )
            .await
            .unwrap();
        assert!(out.is_error);
    }

    #[tokio::test]
    async fn parallel_review_fans_out_lenses() {
        let tool = RunWorkflowTool::new(default_registry(), provider());
        let out = tool
            .execute(
                &ExecutionContext::default(),
                json!({"recipe": "parallel_review", "args": {"target": "fn foo() {}", "lenses": ["a", "b"]}}),
            )
            .await
            .unwrap();
        assert!(!out.is_error, "got: {}", out.content);
        // Both lenses appear as sections, each carrying the agent's output.
        assert!(out.content.contains("### a"), "{}", out.content);
        assert!(out.content.contains("### b"), "{}", out.content);
        assert!(out.content.contains("LENS-OK"), "{}", out.content);
    }

    #[tokio::test]
    async fn parallel_review_requires_target() {
        let tool = RunWorkflowTool::new(default_registry(), provider());
        let out = tool
            .execute(
                &ExecutionContext::default(),
                json!({"recipe": "parallel_review", "args": {}}),
            )
            .await
            .unwrap();
        assert!(out.is_error, "missing target must error");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_event_sink_observes_recipe_internal_agents() {
        use crate::agent::events::{AgentEvent, OnEvent};
        use std::sync::Mutex;

        let captured: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let sink: Arc<OnEvent> = {
            let captured = Arc::clone(&captured);
            Arc::new(move |ev| captured.lock().expect("lock").push(ev))
        };
        let tool = RunWorkflowTool::new(default_registry(), provider()).with_agent_events(sink);
        let out = tool
            .execute(
                &ExecutionContext::default(),
                json!({"recipe": "parallel_review", "args": {"target": "fn foo() {}", "lenses": ["a", "b"]}}),
            )
            .await
            .unwrap();
        assert!(!out.is_error, "got: {}", out.content);

        let events = captured.lock().expect("lock");
        // Each lens agent's full runner lifecycle must reach the sink, keyed by
        // its flow label (the trace's `agent` field).
        for lens in ["a", "b"] {
            let label = format!("review:{lens}");
            assert!(
                events.iter().any(
                    |e| matches!(e, AgentEvent::RunCompleted { agent, .. } if *agent == label)
                ),
                "missing RunCompleted for {label}"
            );
        }
    }

    /// A recipe with two SEQUENTIAL agents — deterministic budget breach: the
    /// first call records its spend, the second is refused admission.
    fn two_step_recipe() -> WorkflowRecipe {
        WorkflowRecipe {
            name: "two_step".into(),
            description: "test recipe".into(),
            args_schema: json!({"type": "object"}),
            run: Arc::new(|ctx, _args| {
                Box::pin(async move {
                    let a = agent(&ctx, "step one").run().await?.unwrap_or_default();
                    let b = agent(&ctx, "step two").run().await?.unwrap_or_default();
                    Ok(format!("{a}+{b}"))
                })
            }),
        }
    }

    /// Fixed text with REAL token usage (so the budget pool actually fills).
    struct CostlyText;
    impl LlmProvider for CostlyText {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text { text: "x".into() }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
                usage: TokenUsage {
                    input_tokens: 800,
                    output_tokens: 200,
                    ..Default::default()
                },
                model: None,
            })
        }
    }

    #[tokio::test]
    async fn budget_arg_breaches_and_surfaces_as_tool_error() {
        let registry = WorkflowRegistry::new().register(two_step_recipe());
        let tool = RunWorkflowTool::new(registry, Arc::new(BoxedProvider::new(CostlyText)));
        let out = tool
            .execute(
                &ExecutionContext::default(),
                json!({"recipe": "two_step", "budget": 100}),
            )
            .await
            .unwrap();
        assert!(out.is_error, "breach must surface: {}", out.content);
        assert!(
            out.content.to_lowercase().contains("budget"),
            "{}",
            out.content
        );
    }

    #[tokio::test]
    async fn without_budget_the_same_recipe_completes() {
        let registry = WorkflowRegistry::new().register(two_step_recipe());
        let tool = RunWorkflowTool::new(registry, Arc::new(BoxedProvider::new(CostlyText)));
        let out = tool
            .execute(&ExecutionContext::default(), json!({"recipe": "two_step"}))
            .await
            .unwrap();
        assert!(!out.is_error, "{}", out.content);
        assert_eq!(out.content, "x+x");
    }

    /// Counts provider calls — the resume test's whole point is that the
    /// SECOND invocation makes ZERO new ones.
    struct CountingText(Arc<std::sync::atomic::AtomicUsize>);
    impl LlmProvider for CountingText {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text { text: "out".into() }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            })
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rerunning_the_same_call_replays_from_the_journal() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let dir = tempfile::tempdir().unwrap();
        let registry = WorkflowRegistry::new().register(two_step_recipe());
        let tool = RunWorkflowTool::new(
            registry,
            Arc::new(BoxedProvider::from_arc(Arc::new(CountingText(
                calls.clone(),
            )))),
        )
        .with_journal_dir(dir.path().to_path_buf());

        let input = json!({"recipe": "two_step", "args": {"k": "v"}});
        let first = tool
            .execute(&ExecutionContext::default(), input.clone())
            .await
            .unwrap();
        assert!(!first.is_error, "{}", first.content);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);

        // Re-invocation: same recipe+args → both agents replay, zero new calls.
        let second = tool
            .execute(&ExecutionContext::default(), input)
            .await
            .unwrap();
        assert!(!second.is_error, "{}", second.content);
        assert_eq!(second.content, first.content, "identical replayed output");
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "the resumed run must make ZERO new provider calls"
        );

        // DIFFERENT args → different journal → fresh run.
        let third = tool
            .execute(
                &ExecutionContext::default(),
                json!({"recipe": "two_step", "args": {"k": "other"}}),
            )
            .await
            .unwrap();
        assert!(!third.is_error);
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            4,
            "changed args must NOT replay the old journal"
        );
    }

    /// A recipe whose single agent asks for worktree isolation.
    fn isolated_recipe() -> WorkflowRecipe {
        WorkflowRecipe {
            name: "isolated".into(),
            description: "test".into(),
            args_schema: json!({"type": "object"}),
            run: Arc::new(|ctx, _args| {
                Box::pin(async move {
                    let out = agent(&ctx, "mutate")
                        .isolation(crate::Isolation::Worktree)
                        .run()
                        .await?
                        .unwrap_or_default();
                    Ok(out)
                })
            }),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn worktree_recipes_are_reachable_through_the_tool() {
        // Fixture repo (worktrees need a HEAD).
        let dir = tempfile::tempdir().unwrap();
        for args in [
            vec!["init", "-q"],
            vec!["config", "user.name", "t"],
            vec!["config", "user.email", "t@t"],
            vec!["commit", "--allow-empty", "-q", "-m", "init"],
        ] {
            assert!(
                std::process::Command::new("git")
                    .current_dir(dir.path())
                    .args(&args)
                    .status()
                    .unwrap()
                    .success()
            );
        }
        let registry = WorkflowRegistry::new().register(isolated_recipe());
        // WITH workspace: the isolated agent runs.
        let tool = RunWorkflowTool::new(registry.clone(), provider())
            .with_workspace(dir.path().to_path_buf());
        let out = tool
            .execute(&ExecutionContext::default(), json!({"recipe": "isolated"}))
            .await
            .unwrap();
        assert!(!out.is_error, "{}", out.content);
        // WITHOUT workspace: the leaf's Config error surfaces honestly.
        let bare = RunWorkflowTool::new(registry, provider());
        let out = bare
            .execute(&ExecutionContext::default(), json!({"recipe": "isolated"}))
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("workspace"), "{}", out.content);
    }

    #[tokio::test]
    async fn without_sink_recipes_still_run() {
        let tool = RunWorkflowTool::new(default_registry(), provider());
        let out = tool
            .execute(
                &ExecutionContext::default(),
                json!({"recipe": "parallel_review", "args": {"target": "x", "lenses": ["a"]}}),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
    }
}
