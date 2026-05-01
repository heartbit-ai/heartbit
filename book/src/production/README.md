# Production Considerations

`heartbit-core` is the framework — agents, tools, providers, memory,
eval. For multi-tenant SaaS deployments, where you want a Kafka-backed
daemon, a dashboard, JWT-scoped tenants, sandboxed bash workspaces,
and a Postgres task store, you compose `heartbit-core` with the
[heartbit umbrella crate](https://crates.io/crates/heartbit) and
deploy through `heartbit-cli`'s `daemon` subcommand. This chapter
points you at the right docs for each axis.

## Sandboxing

Bash execution is the highest-risk tool surface. The umbrella crate's
`sandbox` feature wires Linux landlock so a bash process can only
read and write inside an allowlisted workspace, with environment
variables filtered to a pre-declared set. Apply it via
`BashTool::with_sandbox_policy(policy)` or the workspace-aware
constructor `BashTool::with_sandbox(workspace, env_policy)`.

On non-Linux hosts the sandbox layer is a graceful no-op — the same
code compiles and runs, and policy is enforced by the umbrella where
the kernel supports it. See
[crates/heartbit-cli/README.md](https://github.com/heartbit-ai/heartbit/blob/main/crates/heartbit-cli/README.md)
for the daemon-side wiring of sandbox policies per agent.

## Resource limits

Every production agent needs at least four caps configured: `max_turns`
(stops runaway ReAct loops), `max_tokens` (per-call output cap),
`max_total_tokens` (cumulative input + output budget across all turns;
returns `Error::WithPartialUsage` when exceeded), and `run_timeout_seconds`
(wall-clock deadline for the entire run). Add `max_identical_tool_calls`
to engage doom-loop detection — when an agent repeats the same tool
call N times in a row, subsequent calls get error results instead of
executing. Sensible defaults per agent in production: 25 turns,
8 192 output tokens, a few hundred thousand total tokens, a 5-minute
run deadline, and `max_identical_tool_calls = 3`.

## Observability

Three layers, in increasing weight. First, `OnEvent` callbacks fire
synchronously from the runner — wire one to forward `AgentEvent`
variants (`RunStarted`, `LlmResponse`, `ToolCallStarted`, `RunFailed`,
`GuardrailDenied`, `ModelEscalated`, etc.) into your logging or
metrics system. Second, OpenTelemetry: configure `[telemetry]` in
your `HeartbitConfig` and the `heartbit-cli` binary's
`setup_telemetry` wires an OTLP exporter spanning the agent ReAct
loop. Third, in daemon mode the umbrella exposes Prometheus metrics
and SSE event streams over the Axum HTTP API. See
[docs/platform.md](https://github.com/heartbit-ai/heartbit/blob/main/docs/platform.md)
for the full setup.

## Multi-tenancy

`NamespacedMemory` wraps any `Memory` implementation with a tenant
scope so per-tenant agents read and write only their own entries
without changing the underlying store. The umbrella crate's
`auth/jwt` module validates inbound JWTs against your IdP and binds
the resolved user/tenant to every command and audit record produced
by the daemon. Tenant identity also flows into the agent's system
prompt and into `MemoryEntry::author_tenant_id` for cross-context
attribution. The full multi-tenant deployment guide lives in
[docs/platform.md](https://github.com/heartbit-ai/heartbit/blob/main/docs/platform.md).

## Going beyond library mode

The umbrella's `daemon` feature gives you a Kafka-backed runtime
(commands in, audit + SSE out), a Postgres task store, the
multi-tenant dashboard, the cron scheduler, and the heartbeat pulse
service — everything you need to operate agents as a long-running
service rather than as a library inside another binary. Start with
[crates/heartbit-cli/README.md](https://github.com/heartbit-ai/heartbit/blob/main/crates/heartbit-cli/README.md)
for the binary, then
[docs/platform.md](https://github.com/heartbit-ai/heartbit/blob/main/docs/platform.md)
for the multi-tenant deployment topology.
