//! The `deep_research` workflow recipe: plan → parallel search/read (each
//! angle agent carries its own websearch/webfetch tools) → cross-verify →
//! synthesize a cited report. Born from a live failure (session 6a245538):
//! asked to "deep research", the agent had no harness to route to, its
//! scraped searches died silently, and it fabricated URLs.

use std::sync::Arc;

use serde_json::json;

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
                // Decomposition is cheap classification work → the "fast" role
                // (dynamic model-by-task; degrades to the default without a
                // host factory).
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
                .label("research:plan")
                .model("fast")
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
                let mut dead = 0usize;
                let notes = angles
                    .iter()
                    .zip(results)
                    .map(|(angle, slot)| {
                        let body = slot.flatten().unwrap_or_else(|| {
                            dead += 1;
                            "(angle produced no findings)".to_string()
                        });
                        format!("### Angle: {angle}\n{body}")
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
                if dead == angles.len() {
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
                .label("research:verify")
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
                .label("research:synthesize")
                .run()
                .await?
                .unwrap_or_default();
                if report.trim().is_empty() {
                    return Err(Error::Agent(
                        "deep_research: synthesis produced no report".into(),
                    ));
                }
                Ok(report)
            })
        }),
    }
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

    use crate::BoxedProvider;
    use crate::agent::flow::WorkflowCtx;
    use crate::llm::LlmProvider;
    use crate::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };
    use std::sync::{Arc, Mutex};

    /// Content-routed mock: the parallel stage makes call ORDER
    /// nondeterministic, so responses are selected by prompt substring, never
    /// by call index. Captures every request (shared handle) so the tests can
    /// assert what each stage actually saw — especially the tool wiring.
    struct RoutedProvider {
        captured: Arc<Mutex<Vec<CompletionRequest>>>,
    }

    impl RoutedProvider {
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
            let prompt: String = request
                .messages
                .iter()
                .flat_map(|m| m.content.iter())
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            self.captured.lock().expect("capture lock").push(request);
            // Markers are unique to each stage PROMPT: the verify/synthesize
            // prompts embed the angle OUTPUTS (which contain "FINDINGS:"), so
            // routing must key on prompt-only text like "Research this angle".
            let reply = if prompt.contains("Decompose the question") {
                "1. angle one\n2. angle two"
            } else if prompt.contains("Research this angle") {
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

    async fn run_recipe(
        captured: Arc<Mutex<Vec<CompletionRequest>>>,
        args: serde_json::Value,
    ) -> Result<String, Error> {
        let provider = Arc::new(BoxedProvider::from_arc(Arc::new(RoutedProvider {
            captured,
        })));
        let ctx = WorkflowCtx::builder(provider).build().expect("ctx");
        let r = recipe();
        (r.run)(ctx, args).await
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recipe_happy_path_produces_cited_report() {
        let cap = Arc::new(Mutex::new(Vec::new()));
        let report = run_recipe(
            cap,
            serde_json::json!({"question": "how does plate solving work", "angles": 2}),
        )
        .await
        .unwrap();
        assert!(report.contains("## Sources"), "cited report: {report}");
        assert!(report.contains("ex.org"), "{report}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn angle_agents_carry_web_tools_and_other_stages_do_not() {
        let cap = Arc::new(Mutex::new(Vec::new()));
        let _ = run_recipe(
            cap.clone(),
            serde_json::json!({"question": "q", "angles": 2}),
        )
        .await
        .unwrap();
        let reqs = cap.lock().unwrap();
        assert!(
            reqs.len() >= 5,
            "plan + 2 angles + verify + synthesize, got {}",
            reqs.len()
        );
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
            if prompt.contains("Research this angle") {
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

    #[tokio::test(flavor = "multi_thread")]
    async fn recipe_requires_a_question() {
        let cap = Arc::new(Mutex::new(Vec::new()));
        let err = run_recipe(cap, serde_json::json!({})).await.unwrap_err();
        assert!(err.to_string().contains("question"));
    }

    #[test]
    fn registry_includes_deep_research() {
        let reg = crate::agent::workflow_tool::default_registry();
        assert!(reg.get("deep_research").is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn plan_stage_runs_on_the_fast_model_when_a_factory_exists() {
        use crate::agent::flow::ProviderFactory;

        let cap = Arc::new(Mutex::new(Vec::new()));
        let roles: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(BoxedProvider::from_arc(Arc::new(RoutedProvider {
            captured: cap.clone(),
        })));
        let factory: Arc<ProviderFactory> = {
            let roles = Arc::clone(&roles);
            let provider = Arc::clone(&provider);
            Arc::new(move |role: &str| {
                roles.lock().expect("lock").push(role.to_string());
                // Same routed mock — the test asserts WHICH role was asked for.
                Ok(Arc::clone(&provider))
            })
        };
        let ctx = WorkflowCtx::builder(provider.clone())
            .provider_factory(factory)
            .build()
            .expect("ctx");
        let r = recipe();
        let report = (r.run)(ctx, serde_json::json!({"question": "q", "angles": 2}))
            .await
            .unwrap();
        assert!(report.contains("## Sources"), "{report}");
        let roles = roles.lock().expect("lock");
        assert_eq!(
            roles.as_slice(),
            ["fast"],
            "exactly the PLAN stage resolves through the factory"
        );
    }
}
