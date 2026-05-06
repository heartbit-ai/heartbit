//! Criterion benchmark for `PiiGuardrail::post_llm` over a typical
//! 4 KB assistant response.
//!
//! Validates the Phase 1 cross-cutting fix `T6` (consolidate the 4 PII
//! detectors into a single `RegexSet`) and the impact of moving any
//! per-call `Regex::new` calls behind `LazyLock` (theme `T1`). See
//! `tasks/perf-audit-cross.md` (P-CROSS-2) and
//! `tasks/perf-audit-crosscut.md` (P-XCUT-3).

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use heartbit_core::{
    Guardrail, PiiAction, PiiGuardrail,
    llm::types::{CompletionResponse, ContentBlock, StopReason, TokenUsage},
};

fn sample_response_text() -> String {
    // Mix of clean prose and several PII tokens so the guardrail
    // exercises every detector at least once on every iteration.
    let mut out = String::with_capacity(4096);
    for i in 0..30 {
        out.push_str(&format!(
            "Iteration {i}: contact alice@example.com or +1-415-555-0{i:03} for details. \
             SSN ref 123-45-67{i:02}, card 4111-1111-1111-{i:04}. \
             The agent should refuse to share these details with downstream tools.\n"
        ));
    }
    out
}

fn make_response(text: String) -> CompletionResponse {
    CompletionResponse {
        content: vec![ContentBlock::Text { text }],
        stop_reason: StopReason::EndTurn,
        usage: TokenUsage::default(),
        model: None,
    }
}

fn bench_pii(c: &mut Criterion) {
    let guard = PiiGuardrail::all_builtin(PiiAction::Redact);
    let payload = sample_response_text();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime");

    c.bench_function("guardrail_pii_post_llm_4kb_redact", |b| {
        b.iter(|| {
            let mut response = make_response(payload.clone());
            rt.block_on(async {
                let action = guard.post_llm(&mut response).await.expect("post_llm");
                black_box(action);
                black_box(response.content.len());
            })
        })
    });

    let warn_guard = PiiGuardrail::all_builtin(PiiAction::Warn);
    c.bench_function("guardrail_pii_post_llm_4kb_warn", |b| {
        b.iter(|| {
            let mut response = make_response(payload.clone());
            rt.block_on(async {
                let action = warn_guard.post_llm(&mut response).await.expect("post_llm");
                black_box(action);
            })
        })
    });
}

criterion_group!(benches, bench_pii);
criterion_main!(benches);
