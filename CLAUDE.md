# Heartbit

Multi-agent enterprise runtime in Rust.

## Principles

### TDD Mandatory
- **Write tests FIRST, then implementation.** No exception.
- Red → Green → Refactor cycle for every feature.
- Every public function must have at least one test.
- `cargo test` must pass before any commit.
- Never mark a task complete without proving it works (tests, logs, demo).
- Autonomous bug fixing: given a bug, just fix it. Point at evidence, resolve, zero hand-holding.

### Workflow
- **Plan first**: enter plan mode for any non-trivial task (3+ steps or architectural decisions).
- If something goes sideways, STOP and re-plan — don't keep pushing a broken approach.
- Track progress in plan files with checkable items. Mark complete as you go.
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky, step back and implement the elegant solution.
- Skip elegance checks for simple, obvious fixes — don't over-engineer.

### Rust Quality Gates
```bash
cargo fmt -- --check && cargo clippy -- -D warnings && cargo test
```
All three must pass. No warnings allowed.

### Code Style (from rust-best-practices)
- `thiserror` for library errors (heartbit crate), `anyhow` for application code (CLI).
- Prefer borrowing over cloning. Use `&str` / `impl Into<String>` for parameters.
- Use `?` operator, never `.unwrap()` in library code.
- Iterators over loops. `Vec::with_capacity` for known sizes.
- `pub(crate)` for internal APIs. Keep modules focused.
- Builder pattern for complex configuration.
- No premature abstraction — three similar lines is better than one unused helper.
- **Simplicity first**: every change as simple as possible, touch only what's necessary.

### Architecture
- 3 crates: `heartbit` (lib), `heartbit-cli` (bin), `heartbit-cockpit` (Slint desktop GUI).
- Flat agent hierarchy: orchestrator spawns sub-agents, sub-agents do NOT spawn.
- Three execution paths: standalone (`AgentRunner` + `tokio::JoinSet`), durable (`Restate SDK 0.8`), daemon (Kafka-backed).
- `tokio::JoinSet` for parallel tool execution and sub-agent dispatch (standalone path).
- Restate workflows/services/objects for durable execution with replay (Restate path).
- Daemon mode: Kafka consumer loop, Axum HTTP API, SSE event streaming, cron scheduler.
- MCP Streamable HTTP client for tool server connectivity.
- SSE parser maison for Anthropic streaming (no third-party SSE crate).
- Optional PostgreSQL store for task tracking and audit logging.
- Optional OpenTelemetry tracing via OTLP exporter.
- Workspace dependencies in root Cargo.toml.

### Subagent Strategy
- Use subagents liberally to keep main context window clean.
- Offload research, exploration, and parallel analysis to subagents.
- One focused task per subagent. For complex problems, throw more compute via parallel subagents.

### Self-Improvement
- After ANY correction from the user: update `tasks/lessons.md` with the pattern.
- Write rules that prevent the same mistake from recurring.
- Review lessons at session start for the current project.

### What We Don't Build (Yet)
No NATS, event sourcing, DAG scheduler, gRPC, Redis, Prometheus.
Add when the need arrives, not before.
