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
}

impl RunWorkflowTool {
    /// Build the tool over a registry and the shared provider used to run the
    /// recipe's `flow/` agents.
    pub fn new(registry: WorkflowRegistry, provider: Arc<BoxedProvider>) -> Self {
        Self { registry, provider }
    }
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

        let recipe = self.registry.get(&recipe_name).cloned();
        let provider = self.provider.clone();
        Box::pin(async move {
            let Some(recipe) = recipe else {
                return Ok(ToolOutput::error(format!(
                    "unknown workflow recipe '{recipe_name}'"
                )));
            };
            let ctx = match WorkflowCtx::builder(provider).build() {
                Ok(c) => c,
                Err(e) => return Ok(ToolOutput::error(format!("workflow setup failed: {e}"))),
            };
            match (recipe.run)(ctx, args).await {
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
}
