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
cargo fmt --all -- --check && cargo clippy --workspace --exclude mini-crm --all-targets -- -D warnings && cargo test --workspace --exclude mini-crm
```
All three must pass. No warnings allowed. (`mini-crm` is a WIP app excluded from the
gate — build it directly with `cargo build -p mini-crm`.)

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
- 2 crates: `heartbit` (lib), `heartbit-cli` (bin).
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

### Process Safety
- **NEVER `pkill` or kill running server processes** (heartbit-cloud, heartbit daemon, dashboard dev server, etc.). You will kill the service you need for testing.
- To restart a service: ask the user to restart it, or use a separate terminal.
- If a port is already in use, that means the service is already running — use it, don't kill it.
- Before running `cargo run` for a server: check if it's already running with `curl` to its healthz endpoint first.

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
