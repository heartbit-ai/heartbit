# Heartbit

Multi-agent enterprise runtime in Rust.

## Principles

### TDD Mandatory
- **Write tests FIRST, then implementation.** No exception.
- Red → Green → Refactor cycle for every feature.
- Every public function must have at least one test.
- `cargo test` must pass before any commit.

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

### Architecture
- 2 crates: `heartbit` (lib) and `heartbit-cli` (bin). Split when a module gets too big.
- Flat agent hierarchy: orchestrator spawns sub-agents, sub-agents do NOT spawn.
- `tokio::JoinSet` for parallel tool execution and sub-agent dispatch.
- SSE parser maison for Anthropic streaming (no third-party SSE crate).
- Workspace dependencies in root Cargo.toml.

### What We Don't Build (Yet)
No NATS, event sourcing, DAG scheduler, durable execution, gRPC, Redis, Prometheus, OpenTelemetry.
Add when the need arrives, not before.
