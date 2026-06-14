# 2026 Frontier Implementation — heartbit-core

Goal: implement the 2026 agentic-framework frontier findings, deep-test, fix all, perfect implementation.
Branch: `feat/2026-frontier` (off main / released 2026.613.1).

Honest framing: this is ~8 substantial features (a multi-item roadmap). Delivering each TDD'd + workspace-gate-green, committing incrementally. Status tracked here.

## Roadmap (priority order from the gap analysis)
1. [DONE] Lethal-trifecta tool-exposure analysis + build-time warn (Willison Jun 2025) — `tool/security.rs`, `Tool::security_exposure`, `analyze_tools`, warn in `build()`. 7 tests. Commit 1.
2. [DONE] Dual-LLM / quarantined-content extraction (CaMeL 2503.18813, Design Patterns 2506.08837) — `agent/dual_llm.rs` `QuarantinedReader` (no-tools structural guarantee). 3 tests. Commit 2.
3. [ ] Function-call-hallucination + RAG-groundedness guard (Granite Guardian 2412.07724) — NOTE: tool-name repair + schema validation already in runner; remaining value = a denying guard + groundedness judge. SUBSTANTIAL.
4. [DONE] Cheap→expensive cascade safety guard (Constitutional Classifiers++ 2601.04603, Qwen3Guard-Stream 2510.14276) — `guardrails/cascade.rs` `CascadingGuardrail` (screen short-circuits before LLM judge). 2 tests. Commit 3.
5. [DONE] Verifier-guided test-time compute seam — `agent/verifier.rs` `Verifier`/`select_best`/`LlmVerifier`. 5 tests. Commit 2.
6. [ ] AgentDojo (2406.13352) + InjecAgent (2403.02691) PI-security eval harness. SUBSTANTIAL (scenario infra).
7. [ ] Experience-stage memory: trajectory store + skill auto-acquisition (Voyager-style). SUBSTANTIAL.
8. [ ] Hybrid retrieval (BM25+dense+rerank) + entity-relation GraphRAG. SUBSTANTIAL (embedding wiring).
9. [ ] CONSOLIDATE overlapping orchestration combinators. REFACTOR (risky) — deprioritize.
10. [ ] Full CaMeL Privileged-LLM plan-as-code + capabilities interpreter (the P-LLM half). SUBSTANTIAL.

## STATUS
**4 frontier primitives DELIVERED, TDD'd, workspace-gate-green (5348 tests), committed** on `feat/2026-frontier`:
trifecta analysis, dual-LLM quarantined reader, verifier-guided best-of-N, cascade guardrail.
Remaining (3, 6, 7, 8, 9, 10) are each substantial multi-day features — a genuine roadmap, not a one-pass job. Continuing item-by-item.

## Gate
Full workspace: `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -D warnings && cargo test --workspace`.
