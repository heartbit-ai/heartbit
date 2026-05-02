# Failure-Mode Hardening (B5b)

Three opt-in components that turn the daemon into a fault-tolerant
multi-tenant runtime.

## 1. Idempotency keys

Set the `Idempotency-Key` header on `POST /v1/tasks`:

```http
POST /v1/tasks
Idempotency-Key: dedup-payment-12345

{"task": "process payment 12345"}
```

Subsequent requests with the same `(tenant, key)` pair return the existing
task id without re-executing. Keys do not expire by default — configure
`ttl_hours` to enable the sweep. Once enabled, the daemon's background task
nulls keys older than the TTL.

```toml
[daemon.idempotency]
ttl_hours = 24
sweep_interval_minutes = 60
```

The dedup is scoped to `(tenant_id, idempotency_key)` via a partial unique
index on `daemon_tasks`. Two different tenants supplying the same key create
two separate tasks. The TTL sweep nulls out keys older than `ttl_hours` so
they no longer dedup; the rows themselves are retained for audit purposes.

### Programmatic API

When using `DaemonCommand::SubmitTask` directly (non-HTTP path), set the
`idempotency_key` field:

```rust
use heartbit::daemon::DaemonCommand;

let cmd = DaemonCommand::SubmitTask {
    task: "process payment 12345".into(),
    idempotency_key: Some("dedup-payment-12345".into()),
    // …other fields
};
```

An `Ok(existing_task_id)` is returned without dispatching the task again.

## 2. Per-tenant token cap

Cap concurrent in-flight tokens per tenant:

```toml
[orchestrator]
max_tokens_in_flight_per_tenant = 1_000_000
```

Submissions estimated to push the tenant past the cap fail with
`Error::TenantOverloaded` → HTTP 503 with a `Retry-After: 5` header.
Single-tenant deployments leave this unset (effectively unbounded).

The estimate at submit time is `task_input.len() / 4 + 4096` (rough
chars-to-tokens conversion plus a fixed buffer for tool overhead). The
runner reconciles actual usage per turn during execution, so the in-flight
counter tracks real cost throughout the task's lifetime. On task
completion, the runner releases its cumulative actual tokens — the tenant
returns to its pre-task baseline.

### Capacity planning

| Context window | Typical task estimate |
|---|---|
| 200k (Claude 3.7) | ~50k tokens end-to-end |
| 32k (GPT-4o) | ~8k tokens end-to-end |

A cap of `1_000_000` allows roughly 20 simultaneous 50k-token tasks for a
single tenant. Tune based on your provider's rate limits and the number of
concurrent users you expect per tenant.

## 3. Per-(tenant, provider) circuit breaker

Trip the circuit after N consecutive retry-exhausted attempts. Composes
with the existing `RetryingProvider`:

```toml
[provider.circuit]
failure_threshold = 5
initial_open_duration_seconds = 30
max_open_duration_seconds = 300
backoff_multiplier = 2.0
```

Composition order is `CircuitBreaker<Retrying<Provider>>`. **One permit per
outer call covers a full retry budget.** A `failure_threshold = 5` means
5 consecutive retry-exhausted outer attempts open the circuit — not 5 raw
HTTP failures.

### State machine

```
Closed ──(N consecutive failures)──> Open ──(window expires)──> HalfOpen
  ^                                                                  |
  └──────────────────────(probe succeeds)───────────────────────────┘
                             (probe fails) ──> Open (extended window)
```

- **Closed** — requests pass through. Consecutive failures count.
- **Open** — requests fail fast with `Error::CircuitOpen` for the
  configured duration. After the window, transitions to HalfOpen.
- **HalfOpen** — one request probes the upstream. Success → Closed (counter
  resets). Failure → Open with extended duration (capped at
  `max_open_duration_seconds`).

Note: `backoff_multiplier` is applied to `initial_open_duration` (not to
the previous open duration). With the defaults above, repeated half-open
failures produce a constant `60s` window — not a compounding
`30s → 60s → 120s → 240s` sequence. This is intentional: the
per-(tenant, provider) granularity already bounds blast radius without
requiring true exponential backoff.

### What trips the circuit

The classifier trips on transient errors where waiting can help:

| Error class | Trips circuit? | Why |
|---|---|---|
| `ServerError` (HTTP 5xx) | Yes | Provider outage |
| `RateLimited` (HTTP 429) | Yes | Quota exhausted |
| `Network` (TCP/DNS/TLS/timeout) | Yes | Connectivity |
| `AuthError` (HTTP 401/403) | No | Won't recover from waiting |
| `InvalidRequest` (HTTP 400) | No | Caller bug, not transient |
| `ContextOverflow` | No | Handled by auto-compaction |

Open circuits return `Error::CircuitOpen` immediately — no retries fire
while open. Each `(tenant, provider)` pair has its own circuit, so a
sustained outage at one provider does not impact other providers or other
tenants on the same runtime.

### Observability

Observability of circuit state transitions is not yet instrumented — state
changes are silent today. Consumers can pattern-match on
`Error::CircuitOpen { until, prev_duration }` (returned when permits are
denied) for telemetry at the call site. Per-transition `tracing::` events
inside the breaker itself are a planned follow-up.

Pair with an `AgentEvent::ModelEscalated` listener (from the cascading
provider) if you want to detect provider-level escalation before the
circuit trips.

## Combining all three

All three components compose. A recommended production `daemon-prod.toml`:

```toml
[orchestrator]
max_tokens_in_flight_per_tenant = 1_000_000

[daemon.idempotency]
ttl_hours = 24
sweep_interval_minutes = 60

[provider.circuit]
failure_threshold = 5
initial_open_duration_seconds = 30
max_open_duration_seconds = 300
backoff_multiplier = 2.0
```

With these settings:

- Duplicate client retries deduplicate at the HTTP layer — safe to retry
  `POST /v1/tasks` on timeout without double-charging.
- A noisy tenant cannot starve others by spawning unlimited concurrent tasks.
- A flapping LLM provider trips fast and stays open for a bounded window
  per (tenant, provider) instead of being hammered by retries from every
  queued task.
