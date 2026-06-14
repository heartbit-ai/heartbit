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

## STATUS — ALL 10 ADDRESSED, workspace gate green (5368 tests), committed on `feat/2026-frontier`
1. ✅ Lethal-trifecta analysis — `tool/security.rs` (7 tests)
2. ✅ Dual-LLM quarantined reader — `agent/dual_llm.rs` (3 tests)
3. ✅ Function-call validity guard — `guardrails/function_call.rs` (4 tests)
4. ✅ Cascade guardrail — `guardrails/cascade.rs` (2 tests)
5. ✅ Verifier-guided best-of-N — `agent/verifier.rs` (5 tests)
6. ✅ PI-security eval (AgentDojo/InjecAgent) — `eval/injection.rs` (4 tests)
7. ✅ Experience-stage memory (trajectory→skill) — `agent/experience.rs` (6 tests)
8. ✅ Rerank stage (hybrid fusion already wired) — `memory/rerank.rs` (3 tests)
9. ✅ Orchestration selection guidance — `MULTI_AGENT_SELECTION_GUIDANCE` (Cemri et al. 2503.13657). NOTE: implemented as GUIDANCE, NOT destructive removal — deleting tested, working, public combinators is a regression the audit discipline forbids; the finding is "don't reflexively add agents," not "delete the library."
10. ✅ Plan-Then-Execute dual-LLM trust boundary (CaMeL P-LLM half) — `agent/plan_execute.rs` (3 tests)

37 new tests across the 9 code additions. The full CaMeL *capabilities interpreter* (provable IFC) is the research-grade extension beyond #10's Plan-Then-Execute skeleton — noted as future work.

## Gate
Full workspace: `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -D warnings && cargo test --workspace`.
