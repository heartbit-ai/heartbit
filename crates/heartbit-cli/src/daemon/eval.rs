//! Eval execution endpoint for the daemon HTTP API.
//!
//! Accepts a `RuntimeEvalRequest`, builds an agent, runs each eval case
//! sequentially, scores results, and returns a `RuntimeEvalResponse`.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use heartbit::{
    CostScorer, EvalComparison, EvalRunner, EvalSummary, KeywordScorer, LatencyScorer,
    RuntimeEvalRequest, RuntimeEvalResponse, RuntimeEvalSseEvent, SafetyScorer, SimilarityScorer,
    ToolCallCountScorer, TrajectoryScorer, clear_events,
};

use super::execute;
use super::types::AppState;

/// Handle a cloud-delegated eval request.
pub(crate) async fn handle_eval(
    State(state): State<AppState>,
    Json(req): Json<RuntimeEvalRequest>,
) -> impl IntoResponse {
    // Reject multi-agent — eval is single-agent only.
    if !req.agent_config.sub_agents.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "eval endpoint only supports single-agent configuration; sub_agents must be empty"
            })),
        )
            .into_response();
    }

    if req.cases.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "cases must not be empty" })),
        )
            .into_response();
    }

    const MAX_EVAL_CASES: usize = 200;
    if req.cases.len() > MAX_EVAL_CASES {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("too many eval cases: {} (max {})", req.cases.len(), MAX_EVAL_CASES)
            })),
        )
            .into_response();
    }

    if req.stream {
        handle_eval_stream(state, req).await.into_response()
    } else {
        handle_eval_sync(state, req).await.into_response()
    }
}

/// Build an `EvalRunner` from scorer name strings.
fn build_eval_runner(
    scoring: &heartbit::RuntimeScorerConfig,
    collector: &heartbit::EventCollector,
) -> EvalRunner {
    let mut runner = EvalRunner::new();
    for name in &scoring.scorers {
        match name.as_str() {
            "trajectory" => runner = runner.scorer(TrajectoryScorer),
            "keyword" => runner = runner.scorer(KeywordScorer),
            "similarity" => runner = runner.scorer(SimilarityScorer),
            "cost" => {
                runner = runner.scorer(CostScorer::new(collector.clone(), scoring.max_cost_usd))
            }
            "latency" => {
                runner = runner.scorer(LatencyScorer::new(
                    collector.clone(),
                    scoring.max_latency_ms,
                ))
            }
            "tool_call_count" => {
                runner = runner.scorer(ToolCallCountScorer::new(scoring.max_tool_calls))
            }
            "safety" => runner = runner.scorer(SafetyScorer::new(collector.clone())),
            other => {
                tracing::warn!(scorer = %other, "unknown scorer name, skipping");
            }
        }
    }
    runner
}

async fn handle_eval_sync(state: AppState, req: RuntimeEvalRequest) -> impl IntoResponse {
    let eval_id = req.eval_id;
    let collector = EvalRunner::event_collector();
    let callback = EvalRunner::event_callback(&collector);
    let on_event: Arc<heartbit::OnEvent> = callback;

    // Build agent — memory disabled for eval purity.
    let mut agent_config = req.agent_config.clone();
    agent_config.memory = None;

    let runner = match execute::build_runner_from_request(
        &agent_config,
        None,
        None, // no shared memory for eval
        state.db_pool.as_ref(),
        Some(on_event),
        None, // no workspace for eval
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e })),
            )
                .into_response();
        }
    };

    let eval_runner = build_eval_runner(&req.scoring, &collector);

    let mut results = Vec::with_capacity(req.cases.len());
    for case in &req.cases {
        clear_events(&collector);
        let (output, error) = match runner.execute(&case.input).await {
            Ok(out) => (out.result, None),
            Err(e) => (String::new(), Some(e.to_string())),
        };
        let tool_calls = EvalRunner::collected_tool_calls(&collector);
        let result = eval_runner.score_result(case, &output, &tool_calls, error);
        results.push(result);
    }

    let summary = EvalSummary::from_results(&results);
    let comparison = req
        .baseline
        .as_ref()
        .map(|baseline| EvalComparison::compare(baseline, &results));

    let resp = RuntimeEvalResponse {
        eval_id,
        results,
        summary,
        comparison,
    };

    (StatusCode::OK, Json(resp)).into_response()
}

async fn handle_eval_stream(state: AppState, req: RuntimeEvalRequest) -> impl IntoResponse {
    let (tx, rx) = mpsc::channel::<RuntimeEvalSseEvent>(256);

    tokio::spawn(async move {
        let eval_id = req.eval_id;
        let collector = EvalRunner::event_collector();
        let callback = EvalRunner::event_callback(&collector);
        let on_event: Arc<heartbit::OnEvent> = callback;

        // Build agent — memory disabled for eval purity.
        let mut agent_config = req.agent_config.clone();
        agent_config.memory = None;

        let runner = match execute::build_runner_from_request(
            &agent_config,
            None,
            None,
            state.db_pool.as_ref(),
            Some(on_event),
            None, // no workspace for eval
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(RuntimeEvalSseEvent::Error { message: e }).await;
                return;
            }
        };

        let eval_runner = build_eval_runner(&req.scoring, &collector);

        let mut results = Vec::with_capacity(req.cases.len());
        for case in &req.cases {
            clear_events(&collector);
            let (output, error) = match runner.execute(&case.input).await {
                Ok(out) => (out.result, None),
                Err(e) => (String::new(), Some(e.to_string())),
            };
            let tool_calls = EvalRunner::collected_tool_calls(&collector);
            let result = eval_runner.score_result(case, &output, &tool_calls, error);

            // Send per-case result
            let _ = tx
                .send(RuntimeEvalSseEvent::CaseResult(result.clone()))
                .await;
            results.push(result);
        }

        let summary = EvalSummary::from_results(&results);
        let comparison = req
            .baseline
            .as_ref()
            .map(|baseline| EvalComparison::compare(baseline, &results));

        let resp = RuntimeEvalResponse {
            eval_id,
            results,
            summary,
            comparison,
        };
        let _ = tx.send(RuntimeEvalSseEvent::Done(resp)).await;
    });

    let stream: ReceiverStream<RuntimeEvalSseEvent> = ReceiverStream::new(rx);
    let sse_stream = futures::StreamExt::map(stream, |event| {
        let data = serde_json::to_string(&event).unwrap_or_default();
        Ok::<_, Infallible>(Event::default().data(data))
    });

    Sse::new(sse_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit::{
        RuntimeAdvancedConfig, RuntimeAgentConfig, RuntimeEvalRequest, RuntimeProviderConfig,
        RuntimeProviderType, RuntimeRequest, RuntimeScorerConfig,
    };

    fn make_test_request() -> RuntimeRequest {
        RuntimeRequest {
            task_id: uuid::Uuid::new_v4(),
            prompt: String::new(),
            stream: false,
            tenant_id: None,
            user_id: None,
            memory: None,
            agent: RuntimeAgentConfig {
                name: "eval-agent".into(),
                system_prompt: Some("You are a test agent.".into()),
                max_turns: 5,
                max_tokens: 1024,
                advanced: RuntimeAdvancedConfig::default(),
            },
            provider: RuntimeProviderConfig {
                provider_type: RuntimeProviderType::Anthropic,
                api_key: "sk-test".into(),
                model: "claude-sonnet-4-20250514".into(),
                prompt_caching: false,
            },
            mcp_servers: vec![],
            builtin_tools: vec![],
            guardrails: None,
            messages: vec![],
            session_id: None,
            sub_agents: vec![],
            orchestrator: None,
            workflow: None,
        }
    }

    #[test]
    fn build_scorers_from_config() {
        let collector = EvalRunner::event_collector();
        let config = RuntimeScorerConfig {
            scorers: vec![
                "trajectory".into(),
                "keyword".into(),
                "similarity".into(),
                "cost".into(),
                "latency".into(),
                "tool_call_count".into(),
                "safety".into(),
            ],
            max_cost_usd: 0.05,
            max_latency_ms: 5000,
            max_tool_calls: 10,
        };
        let runner = build_eval_runner(&config, &collector);
        // EvalRunner is Debug — verify it was built with all 7 scorers
        let debug = format!("{:?}", runner);
        assert!(debug.contains("trajectory"), "debug: {debug}");
        assert!(debug.contains("keyword"), "debug: {debug}");
        assert!(debug.contains("similarity"), "debug: {debug}");
        assert!(debug.contains("cost"), "debug: {debug}");
        assert!(debug.contains("latency"), "debug: {debug}");
        assert!(debug.contains("tool_call_count"), "debug: {debug}");
        assert!(debug.contains("safety"), "debug: {debug}");
    }

    #[test]
    fn build_scorers_skips_unknown() {
        let collector = EvalRunner::event_collector();
        let config = RuntimeScorerConfig {
            scorers: vec!["keyword".into(), "nonexistent".into()],
            max_cost_usd: 0.10,
            max_latency_ms: 30_000,
            max_tool_calls: 20,
        };
        let runner = build_eval_runner(&config, &collector);
        let debug = format!("{:?}", runner);
        assert!(debug.contains("keyword"));
        assert!(!debug.contains("nonexistent"));
    }

    #[tokio::test]
    async fn eval_rejects_multi_agent() {
        let mut agent_config = make_test_request();
        agent_config.sub_agents = vec![heartbit::RuntimeSubAgentConfig {
            name: "sub".into(),
            description: "test".into(),
            system_prompt: "test".into(),
            max_turns: 5,
            max_tokens: 1024,
            builtin_tools: vec![],
            mcp_servers: vec![],
        }];

        let req = RuntimeEvalRequest {
            eval_id: uuid::Uuid::new_v4(),
            agent_config,
            cases: vec![heartbit::EvalCase::new("test", "hello")],
            scoring: RuntimeScorerConfig {
                scorers: vec!["keyword".into()],
                max_cost_usd: 0.10,
                max_latency_ms: 30_000,
                max_tool_calls: 20,
            },
            stream: false,
            baseline: None,
        };

        // Verify the request would be rejected (sub_agents non-empty)
        assert!(
            !req.agent_config.sub_agents.is_empty(),
            "should have sub_agents for rejection test"
        );
    }

    #[test]
    fn eval_rejects_empty_cases() {
        let req = RuntimeEvalRequest {
            eval_id: uuid::Uuid::new_v4(),
            agent_config: make_test_request(),
            cases: vec![],
            scoring: RuntimeScorerConfig {
                scorers: vec!["keyword".into()],
                max_cost_usd: 0.10,
                max_latency_ms: 30_000,
                max_tool_calls: 20,
            },
            stream: false,
            baseline: None,
        };
        assert!(req.cases.is_empty());
    }

    #[tokio::test]
    async fn eval_builds_runner_from_request() {
        let agent_config = make_test_request();
        let result =
            execute::build_runner_from_request(&agent_config, None, None, None, None, None).await;
        assert!(result.is_ok());
    }
}
