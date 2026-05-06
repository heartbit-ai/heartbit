//! Criterion benchmark for `InMemoryStore::recall` at N = 1k and N = 10k.
//!
//! Validates Phase 2 algorithmic changes (BM25 inverted index, lazy
//! strength decay, parking_lot lock swap, `Arc<MemoryEntry>` recall
//! return). Compare baselines with `cargo bench --bench memory_recall`
//! before each change and after.
//!
//! See `tasks/perf-audit-memory.md` for the underlying findings.

use chrono::Utc;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use heartbit_core::{
    InMemoryStore, Memory, MemoryEntry, MemoryQuery, MemoryType, auth::TenantScope,
};

fn make_entry(id: u32, agent: &str, content: &str, keywords: &[&str]) -> MemoryEntry {
    MemoryEntry {
        id: format!("entry-{id}"),
        agent: agent.into(),
        content: content.into(),
        category: "fact".into(),
        tags: vec![],
        created_at: Utc::now(),
        last_accessed: Utc::now(),
        access_count: 0,
        importance: 5,
        memory_type: MemoryType::Episodic,
        keywords: keywords.iter().map(|s| s.to_string()).collect(),
        summary: None,
        strength: 1.0,
        related_ids: vec![],
        source_ids: vec![],
        confidentiality: Default::default(),
        embedding: None,
        author_user_id: None,
        author_tenant_id: None,
    }
}

fn populate(store: &InMemoryStore, n: u32, scope: &TenantScope) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime");
    rt.block_on(async {
        for i in 0..n {
            // Spread agents and keywords so recall actually has work to do.
            let agent = match i % 5 {
                0 => "researcher",
                1 => "planner",
                2 => "executor",
                3 => "reviewer",
                _ => "critic",
            };
            let content = format!(
                "this is memory entry {i} about rust performance optimization, async runtimes, and tokio internals"
            );
            let keywords = ["rust", "performance", "tokio", "async", "memory"];
            store
                .store(scope, make_entry(i, agent, &content, &keywords))
                .await
                .expect("store");
        }
    });
}

fn bench_recall(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_recall");
    let scope = TenantScope::default();

    for &size in &[1_000u32, 10_000u32] {
        let store = InMemoryStore::new();
        populate(&store, size, &scope);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("text_query_top10", size), &size, |b, _| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            b.iter(|| {
                rt.block_on(async {
                    let query = MemoryQuery {
                        text: Some("performance tokio async".into()),
                        limit: 10,
                        reinforce: false,
                        ..Default::default()
                    };
                    let results = store.recall(&scope, query).await.expect("recall");
                    black_box(results.len())
                })
            })
        });

        group.bench_with_input(
            BenchmarkId::new("agent_filter_top10", size),
            &size,
            |b, _| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap();
                b.iter(|| {
                    rt.block_on(async {
                        let query = MemoryQuery {
                            agent: Some("researcher".into()),
                            limit: 10,
                            reinforce: false,
                            ..Default::default()
                        };
                        let results = store.recall(&scope, query).await.expect("recall");
                        black_box(results.len())
                    })
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_recall);
criterion_main!(benches);
