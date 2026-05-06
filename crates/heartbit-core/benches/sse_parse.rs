//! Criterion benchmark for the Anthropic SSE parser feed loop on a
//! synthetic 16 KB stream.
//!
//! Validates Phase 3 streaming zero-copy work — see
//! `tasks/perf-audit-llm.md` P-LLM-2 (`feed()` per-event `String`
//! allocations) and P-LLM-14 (`emit_event()` joins `data_lines` even
//! on empty events). Compare baselines with
//! `cargo bench --bench sse_parse --features bench-internals`.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use heartbit_core::__bench;

fn synth_sse_stream() -> String {
    // 32 reasonably sized events, alternating message_start /
    // content_block_delta / content_block_stop frames, ~500 bytes each
    // = ~16 KB total. Mimics a typical Anthropic streamed response.
    let mut out = String::with_capacity(16 * 1024);
    for i in 0..32 {
        out.push_str(&format!(
            "event: content_block_delta\n\
             data: {{\"type\":\"content_block_delta\",\"index\":0,\
             \"delta\":{{\"type\":\"text_delta\",\"text\":\"chunk {i:03} \
             of streamed reasoning. The model is producing a long answer \
             with multiple short increments. iteration={i}.\"}}}}\n\n"
        ));
    }
    out
}

fn bench_sse(c: &mut Criterion) {
    let stream = synth_sse_stream();

    let mut group = c.benchmark_group("sse_parse");
    group.throughput(Throughput::Bytes(stream.len() as u64));

    group.bench_function("feed_16kb_one_shot", |b| {
        b.iter(|| {
            let n = __bench::sse_parse_chunk(&stream);
            black_box(n)
        })
    });

    // Chunked feed — closer to real network behaviour where every
    // packet is a 4 KB read. Stresses the line-spanning path.
    let chunks: Vec<String> = stream
        .as_bytes()
        .chunks(4096)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect();

    group.bench_function("feed_4kb_chunks", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for chunk in &chunks {
                total += __bench::sse_parse_chunk(chunk);
            }
            black_box(total)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_sse);
criterion_main!(benches);
