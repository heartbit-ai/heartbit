//! LIVE demo: the code-generation harness ([`CodeAgentBuilder`]) drives `qwen3-235b`
//! to make a pre-written, initially-failing test pass — and the **`verify` tool's
//! exit code** (not the model's claim) gates completion via [`code_goal`].
//!
//! This is the code analogue of `agent/tetris_live.rs`: there the browser harness
//! verified a built artifact; here the verify tool verifies generated code. Because
//! a test's exit code is deterministic ground truth, the final assertion re-runs the
//! test INDEPENDENTLY (an oracle) rather than trusting the LLM judge.
//!
//! ```text
//! OPENROUTER_API_KEY=sk-or-... cargo test -p heartbit-core --lib \
//!   codegen::live::live_codegen_harness_solves_and_verifies_qwen -- --ignored --nocapture
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::agent::events::{AgentEvent, OnEvent};
use crate::codegen::{CodeAgentBuilder, code_goal};
use crate::llm::BoxedProvider;

const MODEL: &str = "qwen/qwen3-235b-a22b-2507";

/// A pre-written test the agent must satisfy. Plain `assert` + `python3 <file>` so
/// there is no test-framework dependency; exit 0 iff every assertion holds.
const TEST_PY: &str = r#"from solution import is_palindrome

assert is_palindrome("A man, a plan, a canal: Panama") is True
assert is_palindrome("race a car") is False
assert is_palindrome("") is True
assert is_palindrome("0P") is False
assert is_palindrome(".,") is True
print("ALL TESTS PASSED")
"#;

const TASK: &str = "There is a failing test file `test_solution.py` in the workspace. Create \
`solution.py` so the test passes. It must define `is_palindrome(s)` that returns True iff `s` \
reads the same forwards and backwards considering ONLY alphanumeric characters and ignoring \
case. Run the `verify` tool until it reports VERIFY_RESULT: PASS, then stop.";

#[tokio::test]
#[ignore = "live: needs OpenRouter key + python3 + network"]
async fn live_codegen_harness_solves_and_verifies_qwen() {
    let key = std::env::var("OPENROUTER_API_KEY")
        .or_else(|_| std::env::var("LLM_API_KEY"))
        .expect("set OPENROUTER_API_KEY to run this live test");

    let worker = Arc::new(crate::OpenRouterProvider::new(key.clone(), MODEL));
    let judge = Arc::new(BoxedProvider::new(crate::OpenRouterProvider::new(
        key, MODEL,
    )));

    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("test_solution.py"), TEST_PY).expect("write test");

    let turns = Arc::new(AtomicUsize::new(0));
    let tool_log = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let turn_sink = Arc::clone(&turns);
    let log_sink = Arc::clone(&tool_log);
    let on_event: Arc<OnEvent> = Arc::new(move |e: AgentEvent| match e {
        AgentEvent::TurnStarted { turn, .. } => {
            turn_sink.fetch_max(turn, Ordering::SeqCst);
        }
        AgentEvent::ToolCallCompleted {
            tool_name,
            is_error,
            duration_ms,
            ..
        } => {
            log_sink.lock().unwrap().push(format!(
                "{:>6}ms {} {}",
                duration_ms,
                if is_error { "ERR" } else { "ok " },
                tool_name
            ));
        }
        _ => {}
    });

    let agent = CodeAgentBuilder::new(worker, dir.path())
        .name("coder")
        .verify_command("python3 test_solution.py")
        .max_turns(24)
        .max_identical_tool_calls(3)
        .on_event(on_event)
        .goal(code_goal(judge))
        .build()
        .expect("build code harness");

    let out = agent
        .execute(TASK)
        .await
        .expect("code agent run should succeed");

    // ---- INDEPENDENT deterministic oracle: re-run the test ourselves. The exit
    // ---- code is the ground truth, regardless of what the LLM judge concluded.
    let oracle = std::process::Command::new("python3")
        .arg("test_solution.py")
        .current_dir(dir.path())
        .output()
        .expect("run python3 oracle");
    let oracle_pass = oracle.status.success();

    eprintln!("\n=== live_codegen_harness_solves_and_verifies_qwen ===");
    eprintln!("workspace        : {}", dir.path().display());
    eprintln!("turns            : {}", turns.load(Ordering::SeqCst));
    eprintln!("tool calls       : {}", out.tool_calls_made);
    eprintln!("goal_met (judge) : {:?}", out.goal_met);
    eprintln!("tokens           : {:?}", out.tokens_used);
    eprintln!("--- tool log ---");
    for line in tool_log.lock().unwrap().iter() {
        eprintln!("  {line}");
    }
    eprintln!(
        "ORACLE (independent python3 test_solution.py): {}",
        if oracle_pass { "PASS (exit 0)" } else { "FAIL" }
    );
    eprintln!("agent answer     : {}", out.result.trim());

    // The agent produced the file...
    assert!(
        dir.path().join("solution.py").exists(),
        "the agent must have written solution.py"
    );
    // ...and a goal ran...
    assert!(out.goal_met.is_some(), "a completion goal was set");
    // ...and the DETERMINISTIC oracle confirms the generated code actually passes.
    assert!(
        oracle_pass,
        "independent oracle: python3 test_solution.py must pass.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&oracle.stdout),
        String::from_utf8_lossy(&oracle.stderr),
    );
}
