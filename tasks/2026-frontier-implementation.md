# 2026 Frontier Implementation — heartbit-core

Goal: implement the 2026 agentic-framework frontier findings, deep-test, fix all, perfect implementation.
Branch: `feat/2026-frontier` (off main / released 2026.613.1).

Honest framing: this is ~8 substantial features (a multi-item roadmap). Delivering each TDD'd + workspace-gate-green, committing incrementally. Status tracked here.

## Roadmap (priority order from the gap analysis)
1. [DONE] Lethal-trifecta tool-exposure analysis + build-time warn (Willison Jun 2025) — `tool/security.rs`, `Tool::security_exposure`, `analyze_tools`, warn in `build()`. 7 tests.
2. [IN PROGRESS] Dual-LLM / quarantined-content extraction (CaMeL 2503.18813, Design Patterns 2506.08837) — the #1 PI-defense differentiator.
3. [ ] Function-call-hallucination + RAG-groundedness guard (Granite Guardian 2412.07724).
4. [ ] Streaming + cheap→expensive cascade safety guard (Constitutional Classifiers++ 2601.04603, Qwen3Guard-Stream 2510.14276).
5. [ ] Verifier-guided test-time compute seam (PRM / best-of-N with verifier).
6. [ ] AgentDojo (2406.13352) + InjecAgent (2403.02691) PI-security eval harness.
7. [ ] Experience-stage memory: trajectory store + skill auto-acquisition (Voyager-style).
8. [ ] Hybrid retrieval (BM25+dense+rerank) + entity-relation GraphRAG.
9. [ ] CONSOLIDATE overlapping orchestration combinators behind the flow engine.

## Gate
Full workspace: `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -D warnings && cargo test --workspace`.
