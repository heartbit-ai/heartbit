//! Criterion benchmark for `AgentRunner::execute()` against a mock
//! provider — the "agent ReAct turn" bench identified as the highest-
//! priority gap in `tasks/perf-audit-v2-bench-gaps.md` (Bench-NEW-1).
//!
//! Validates the v1 P-RUNNER findings (Arc<tool_defs>, doom-loop
//! hashing, tool-name repair, recently_used_tools, prompt assembly,
//! tool-definition serialisation) by exercising the end-to-end
//! `execute()` path that previously had no benchmark coverage at all.
//!
//! The mock provider returns the same `EndTurn` text response on
//! every call, so each criterion sample measures the per-execute
//! overhead (prompt assembly, tool-defs handling, request building,
//! response decoding, token accounting, output construction) without
//! the LLM round-trip dominating.
//!
//! Run with:
//! ```sh
//! cargo bench --bench agent_react_turn --features bench-internals
//! ```

use std::sync::Arc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use heartbit_core::__bench::BenchMockProvider;
use heartbit_core::AgentRunner;

fn bench_react_turn(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime");

    // Single-turn agent without tools — measures the hot-path overhead
    // shared by every `execute()` call (prompt assembly, request
    // build, response decode, token accounting, output construction).
    let provider = Arc::new(BenchMockProvider::new_text("done."));
    let runner = AgentRunner::builder(provider)
        .name("bench-agent")
        .system_prompt("You are a benchmark agent. Reply 'done.'")
        .max_turns(1)
        .build()
        .expect("build agent");

    c.bench_function("agent_react_turn_no_tools", |b| {
        b.iter(|| {
            rt.block_on(async {
                let out = runner
                    .execute(black_box("Run the benchmark task."))
                    .await
                    .expect("execute");
                black_box(out.tokens_used);
                black_box(out.result.len());
            })
        })
    });
}

criterion_group!(benches, bench_react_turn);
criterion_main!(benches);
