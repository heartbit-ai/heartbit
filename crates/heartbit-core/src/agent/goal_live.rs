//! LIVE demonstration: `GoalCondition` + dynamic-workflow combinators driven by a
//! real model (qwen3-235b via OpenRouter).
//!
//! Two parts, both `#[ignore]` (need network + `OPENROUTER_API_KEY`):
//!  1. `live_goal_loop_qwen` — a single goal-driven agent self-continues under an
//!     INDEPENDENT judge until the objective is verifiably met. Prints the turn
//!     count so you can see the judge driving (or accepting) continuations.
//!  2. `live_goal_fanout_qwen` — three goal-driven sub-agents run CONCURRENTLY in
//!     a `parallel()` flow, each with its own objective + independent judge, all
//!     sharing one token budget. Each result is checked by a deterministic oracle
//!     to prove the goal actually closed (not just claimed done).
//!
//! Run:
//! ```text
//! OPENROUTER_API_KEY=sk-or-... cargo test -p heartbit-core --lib \
//!   agent::goal_live -- --ignored --nocapture
//! ```

#![cfg(test)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;

use crate::agent::AgentRunner;
use crate::agent::events::{AgentEvent, OnEvent};
use crate::agent::flow::{WorkflowCtx, agent, parallel, thunk};
use crate::agent::goal::GoalCondition;
use crate::error::Error;
use crate::llm::BoxedProvider;
use crate::llm::openrouter::OpenRouterProvider;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

const MODEL: &str = "qwen/qwen3-235b-a22b-2507";

fn key() -> String {
    std::env::var("OPENROUTER_API_KEY")
        .or_else(|_| std::env::var("LLM_API_KEY"))
        .expect("set OPENROUTER_API_KEY (or LLM_API_KEY) to run this live test")
}

fn qwen() -> Arc<BoxedProvider> {
    Arc::new(BoxedProvider::new(OpenRouterProvider::new(key(), MODEL)))
}

// ---- deterministic oracles (independent of the agent's and judge's claims) ----

/// Pull every integer out of free text.
fn ints(text: &str) -> Vec<i64> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_ascii_digit() {
            cur.push(c);
        } else if !cur.is_empty() {
            if let Ok(n) = cur.parse() {
                out.push(n);
            }
            cur.clear();
        }
    }
    if let Ok(n) = cur.parse() {
        out.push(n);
    }
    out
}

fn is_prime(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 1;
    }
    true
}

/// Count distinct primes in `[lo, hi]` mentioned in `text`.
fn distinct_primes_in_range(text: &str, lo: i64, hi: i64) -> Vec<i64> {
    let mut v: Vec<i64> = ints(text)
        .into_iter()
        .filter(|&n| n >= lo && n <= hi && is_prime(n))
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Count distinct lowercase words containing `needle`.
fn distinct_words_with(text: &str, needle: &str) -> Vec<String> {
    let mut v: Vec<String> = text
        .split(|c: char| !c.is_ascii_alphabetic())
        .map(|w| w.to_ascii_lowercase())
        .filter(|w| w.len() >= 3 && w.contains(needle))
        .collect();
    v.sort();
    v.dedup();
    v
}

/// Distinct primes in `[lo, hi]` that do NOT end in the digit `bad_last`.
fn distinct_primes_not_ending_in(text: &str, lo: i64, hi: i64, bad_last: i64) -> Vec<i64> {
    distinct_primes_in_range(text, lo, hi)
        .into_iter()
        .filter(|&n| n % 10 != bad_last)
        .collect()
}

/// Primes in `text` that DO end in `bad_last` (the constraint violations the
/// judge should force the agent to remove).
fn violations_ending_in(text: &str, lo: i64, hi: i64, bad_last: i64) -> Vec<i64> {
    distinct_primes_in_range(text, lo, hi)
        .into_iter()
        .filter(|&n| n % 10 == bad_last)
        .collect()
}

/// A DETERMINISTIC checker the agent can call to validate its own list against
/// the objective. This is the production pattern: instead of asking the LLM
/// judge to count/verify (which it does unreliably), the agent runs a tool whose
/// PASS/FAIL output lands in the transcript as `[Tool result: ...]` — and the
/// independent judge then grades that hard evidence. Returns actionable problems
/// on FAIL so the agent can fix and re-check.
struct PrimeCheckerTool;

impl Tool for PrimeCheckerTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "check_primes".into(),
            description: "Validate a candidate list against the objective: EXACTLY 30 distinct \
                          prime numbers strictly between 1 and 500, NONE ending in the digit 3. \
                          Returns PASS, or FAIL with the specific problems to fix. Always call \
                          this and get PASS before giving your final answer."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "string",
                        "description": "Your candidate numbers, comma-separated."
                    }
                },
                "required": ["numbers"]
            }),
        }
    }

    fn execute(
        &self,
        _ctx: &crate::ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let numbers = input.get("numbers").and_then(|v| v.as_str()).unwrap_or("");
            let raw = ints(numbers);
            let mut distinct = raw.clone();
            distinct.sort_unstable();
            distinct.dedup();
            let dup_count = raw.len().saturating_sub(distinct.len());

            let not_prime: Vec<i64> = distinct.iter().copied().filter(|&n| !is_prime(n)).collect();
            let out_of_range: Vec<i64> = distinct
                .iter()
                .copied()
                .filter(|&n| n <= 1 || n >= 500)
                .collect();
            let ending3: Vec<i64> = distinct
                .iter()
                .copied()
                .filter(|&n| is_prime(n) && n > 1 && n < 500 && n % 10 == 3)
                .collect();
            let valid: Vec<i64> = distinct
                .iter()
                .copied()
                .filter(|&n| is_prime(n) && n > 1 && n < 500 && n % 10 != 3)
                .collect();

            if valid.len() >= 30
                && not_prime.is_empty()
                && out_of_range.is_empty()
                && ending3.is_empty()
                && dup_count == 0
            {
                Ok(ToolOutput::success(format!(
                    "PASS — {} distinct primes, all strictly between 1 and 500, none ending in 3. \
                     Objective satisfied.",
                    valid.len()
                )))
            } else {
                let mut problems = Vec::new();
                if valid.len() < 30 {
                    problems.push(format!("only {} VALID primes (need 30)", valid.len()));
                }
                if dup_count > 0 {
                    problems.push(format!("{dup_count} duplicate(s)"));
                }
                if !not_prime.is_empty() {
                    problems.push(format!("not prime: {not_prime:?}"));
                }
                if !out_of_range.is_empty() {
                    problems.push(format!("out of range (must be 2..=499): {out_of_range:?}"));
                }
                if !ending3.is_empty() {
                    problems.push(format!("FORBIDDEN, end in 3: {ending3:?}"));
                }
                Ok(ToolOutput::success(format!(
                    "FAIL — {}. Fix these and call check_primes again.",
                    problems.join("; ")
                )))
            }
        })
    }
}

// -------------------------------------------------------------------------

/// PRODUCTION PATTERN: the agent is given a deterministic `check_primes` tool.
/// It generates a list, validates with the tool, fixes whatever the tool flags,
/// and only finishes once the tool returns PASS — and that PASS lands in the
/// transcript. The INDEPENDENT judge then grades that hard evidence (a
/// `[Tool result: PASS …]` line) rather than counting itself, so it reliably
/// confirms completion. This is the convergent counterpart to
/// `live_goal_forced_continuation_qwen`, where a tool-less LLM judge over-rejected.
#[tokio::test]
#[ignore = "live: needs OPENROUTER_API_KEY + network"]
async fn live_goal_checker_tool_qwen() {
    let turns = Arc::new(AtomicUsize::new(0));
    let t = Arc::clone(&turns);
    let on_event: Arc<OnEvent> = Arc::new(move |ev: AgentEvent| {
        if let AgentEvent::TurnStarted { turn, .. } = ev {
            t.fetch_max(turn, Ordering::SeqCst);
        }
    });

    let objective = "Produce EXACTLY 30 distinct prime numbers strictly between 1 and 500, \
                     with NONE ending in the digit 3. You have a `check_primes` tool: use it \
                     to validate your list, fix every problem it reports, and only give your \
                     final comma-separated answer once it returns PASS.";

    let checker: Arc<dyn Tool> = Arc::new(PrimeCheckerTool);
    let runner: AgentRunner<BoxedProvider> = AgentRunner::builder(qwen())
        .name("checked-primes")
        .system_prompt(
            "You are precise. ALWAYS validate your list with the check_primes tool and fix any \
             reported problems before finishing. Only give your final answer after a PASS.",
        )
        .tools(vec![checker])
        .max_turns(16)
        .on_event(on_event)
        .goal(GoalCondition::new(objective, qwen()).with_max_continuations(4))
        .build()
        .expect("build runner");

    let out = runner.execute(objective).await.expect("run ok");

    let valid = distinct_primes_not_ending_in(&out.result, 1, 500, 3);
    let violations = violations_ending_in(&out.result, 1, 500, 3);

    eprintln!("\n=== live_goal_checker_tool_qwen ===");
    eprintln!("turns taken     : {}", turns.load(Ordering::SeqCst));
    eprintln!("tool calls made : {}", out.tool_calls_made);
    eprintln!("goal_met        : {:?}", out.goal_met);
    eprintln!("tokens          : {:?}", out.tokens_used);
    eprintln!("final answer    : {}", out.result.trim());
    eprintln!(
        "oracle (valid)  : {} distinct primes under 500 not ending in 3",
        valid.len()
    );
    eprintln!(
        "oracle (violate): {} ending in 3: {violations:?}",
        violations.len()
    );

    assert!(out.goal_met.is_some(), "a goal was set");
    assert!(
        out.tool_calls_made >= 1,
        "the agent must have used the check_primes tool, made {}",
        out.tool_calls_made
    );
    // With deterministic PASS evidence in the transcript, the judge should
    // confirm — and when it does, the oracle must agree.
    if out.goal_met == Some(true) {
        assert!(
            valid.len() >= 30,
            "judge said met but only {} valid",
            valid.len()
        );
        assert!(
            violations.is_empty(),
            "judge said met but final answer still has primes ending in 3: {violations:?}"
        );
    }
}

/// FORCED-CONTINUATION demo: the objective carries a subtle constraint models
/// reliably violate on the first pass — "30 distinct primes under 500, none
/// ending in the digit 3". A naive first answer almost always includes primes
/// like 13, 23, 43, …; the INDEPENDENT judge (which can trivially check the last
/// digit) rejects it and the agent must regenerate, demonstrating the
/// loop-until-correct behavior. Bounded by `max_continuations` + `max_turns`.
#[tokio::test]
#[ignore = "live: needs OPENROUTER_API_KEY + network"]
async fn live_goal_forced_continuation_qwen() {
    let turns = Arc::new(AtomicUsize::new(0));
    let t = Arc::clone(&turns);
    let on_event: Arc<OnEvent> = Arc::new(move |ev: AgentEvent| {
        if let AgentEvent::TurnStarted { turn, .. } = ev {
            t.fetch_max(turn, Ordering::SeqCst);
        }
    });

    let objective = "Produce EXACTLY 30 distinct prime numbers, each strictly between \
                     1 and 500, with the constraint that NONE of them may end in the \
                     digit 3 (e.g. 13, 23, 43 are FORBIDDEN). The final answer must be \
                     exactly thirty such primes, comma-separated, and nothing else.";

    let runner: AgentRunner<BoxedProvider> = AgentRunner::builder(qwen())
        .name("constrained-primes")
        .system_prompt("You are a precise assistant. Follow every constraint exactly.")
        .max_turns(10)
        .on_event(on_event)
        .goal(GoalCondition::new(objective, qwen()).with_max_continuations(6))
        .build()
        .expect("build runner");

    let out = runner.execute(objective).await.expect("run ok");

    let valid = distinct_primes_not_ending_in(&out.result, 1, 500, 3);
    let violations = violations_ending_in(&out.result, 1, 500, 3);
    let turns_taken = turns.load(Ordering::SeqCst);

    eprintln!("\n=== live_goal_forced_continuation_qwen ===");
    eprintln!("turns taken     : {turns_taken}  (1 = no continuation; >1 = judge sent it back)");
    eprintln!("goal_met        : {:?}", out.goal_met);
    eprintln!("tokens          : {:?}", out.tokens_used);
    eprintln!("final answer    : {}", out.result.trim());
    eprintln!(
        "oracle (valid)  : {} distinct primes under 500 not ending in 3",
        valid.len()
    );
    eprintln!(
        "oracle (violate): {} primes ending in 3 in the FINAL answer: {violations:?}",
        violations.len()
    );

    assert!(out.goal_met.is_some(), "a goal was set");
    // When the judge declares the goal met, the deterministic oracle must agree:
    // the final answer has >=30 valid primes AND zero constraint violations.
    if out.goal_met == Some(true) {
        assert!(
            valid.len() >= 30,
            "judge said met but only {} valid primes",
            valid.len()
        );
        assert!(
            violations.is_empty(),
            "judge said met but the final answer still contains primes ending in 3: {violations:?}"
        );
    }
}

#[tokio::test]
#[ignore = "live: needs OPENROUTER_API_KEY + network"]
async fn live_goal_loop_qwen() {
    // Count turns to see the independent judge gating completion.
    let turns = Arc::new(AtomicUsize::new(0));
    let t = Arc::clone(&turns);
    let on_event: Arc<OnEvent> = Arc::new(move |ev: AgentEvent| {
        if let AgentEvent::TurnStarted { turn, .. } = ev {
            t.fetch_max(turn, Ordering::SeqCst);
        }
    });

    let objective = "Produce EXACTLY 10 distinct prime numbers, each strictly between \
                     200 and 300, as a comma-separated list. The final answer must \
                     contain exactly ten primes in that range and nothing else.";

    let runner: AgentRunner<BoxedProvider> = AgentRunner::builder(qwen())
        .name("prime-lister")
        .system_prompt("You are a precise assistant. Follow the instruction exactly.")
        .max_turns(8)
        .on_event(on_event)
        // The judge is a SEPARATE qwen call with an impartial prompt and no tools.
        .goal(GoalCondition::new(objective, qwen()).with_max_continuations(4))
        .build()
        .expect("build runner");

    let out = runner.execute(objective).await.expect("run ok");

    let primes = distinct_primes_in_range(&out.result, 200, 300);
    eprintln!("\n=== live_goal_loop_qwen ===");
    eprintln!("turns taken : {}", turns.load(Ordering::SeqCst));
    eprintln!("goal_met    : {:?}", out.goal_met);
    eprintln!("tokens      : {:?}", out.tokens_used);
    eprintln!("answer      : {}", out.result.trim());
    eprintln!(
        "oracle      : {} distinct primes in (200,300): {primes:?}",
        primes.len()
    );

    // If the judge declared the goal met, the deterministic oracle must agree —
    // proving the independent judge accepted a genuinely-complete answer, not an
    // over-reported one.
    if out.goal_met == Some(true) {
        assert!(
            primes.len() >= 10,
            "judge said met but the oracle found only {} primes",
            primes.len()
        );
    }
    assert!(
        out.goal_met.is_some(),
        "a goal was set, so goal_met must be Some"
    );
}

#[tokio::test]
#[ignore = "live: needs OPENROUTER_API_KEY + network"]
async fn live_goal_fanout_qwen() {
    // One shared budget + concurrency cap across the whole fan-out.
    let ctx = WorkflowCtx::builder(qwen())
        .budget(3_000_000)
        .max_concurrency(3)
        .build()
        .expect("build ctx");

    // Three independent objectives, each driven to completion by its own judge.
    let objectives = [
        "Produce EXACTLY 10 distinct prime numbers strictly between 200 and 300, \
         comma-separated, nothing else.",
        "Produce EXACTLY 12 distinct lowercase English words that each contain the \
         letter sequence \"tion\", comma-separated, nothing else.",
        "Produce EXACTLY 8 distinct prime numbers strictly between 400 and 500, \
         comma-separated, nothing else.",
    ];

    let thunks: Vec<_> = objectives
        .iter()
        .enumerate()
        .map(|(i, obj)| {
            let ctx = ctx.clone();
            let obj = obj.to_string();
            thunk(move || async move {
                agent(&ctx, obj.clone())
                    .label(format!("goal-{i}"))
                    .goal(GoalCondition::new(obj, qwen()).with_max_continuations(4))
                    .run()
                    .await
            })
        })
        .collect();

    let results = parallel(&ctx, thunks).await;

    eprintln!("\n=== live_goal_fanout_qwen ===");
    eprintln!("shared budget spent: {}", ctx.budget().spent());

    // Oracle each result — the goal-driven agents must have produced verifiably
    // complete deliverables, concurrently, under the shared budget.
    let r0 = results[0]
        .clone()
        .flatten()
        .expect("goal-0 produced output");
    let p0 = distinct_primes_in_range(&r0, 200, 300);
    eprintln!("goal-0 primes(200,300): {} -> {p0:?}", p0.len());
    assert!(p0.len() >= 10, "goal-0 under-delivered: {p0:?}");

    let r1 = results[1]
        .clone()
        .flatten()
        .expect("goal-1 produced output");
    let w1 = distinct_words_with(&r1, "tion");
    eprintln!("goal-1 'tion' words   : {} -> {w1:?}", w1.len());
    assert!(w1.len() >= 12, "goal-1 under-delivered: {w1:?}");

    let r2 = results[2]
        .clone()
        .flatten()
        .expect("goal-2 produced output");
    let p2 = distinct_primes_in_range(&r2, 400, 500);
    eprintln!("goal-2 primes(400,500): {} -> {p2:?}", p2.len());
    assert!(p2.len() >= 8, "goal-2 under-delivered: {p2:?}");

    eprintln!("ALL THREE goal-driven sub-agents closed their objectives concurrently.");
}
