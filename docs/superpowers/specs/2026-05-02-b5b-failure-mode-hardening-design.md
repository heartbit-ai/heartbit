# B5b — Failure-Mode Hardening: Idempotency, Per-Tenant Memory Isolation, Circuit Breakers

**Date:** 2026-05-02
**Status:** Design — pending user approval before implementation plan
**Scope:** `crates/heartbit-core/src/{agent,llm}/`, `crates/heartbit/src/{daemon,store}/`, `crates/heartbit-cli/src/main.rs`, Postgres schema migration
**Estimated effort:** ~5–7 working days, ~3 implementation commits + verification + docs commit.
**Public API breakage:** Pre-release additive changes only. `DaemonCommand::SubmitTask` gains an optional `idempotency_key` field with `#[serde(default)]` — existing serialized payloads still deserialize. The `tasks` table gets a new `tenant_id TEXT NOT NULL DEFAULT ''` column (additive, default-backfilled). New types (`TenantTokenTracker`, `ProviderCircuit`, `CircuitTracker`, `CircuitBreakerProvider`) are additive. No existing public API changes.

## Background

The 2026-04-30 production-readiness review identified three failure modes deferred from B4:

1. **Kafka redelivery causes duplicate execution.** `DaemonCommand::SubmitTask` has no idempotency key. A Kafka redelivery (or HTTP client retry) re-runs the agent — paying for the LLM call twice, sending duplicate Telegram messages, double-writing to memory and audit. The HTTP-only daemon path is exposed too: clients with naive retry-on-timeout produce the same problem.

2. **Auto-compaction is global; one tenant can OOM the daemon.** The current in-memory token tracker is a process-wide structure. A multi-tenant deployment serving 100 tenants where one tenant fires a 10M-token agent run starves the tracker for all other tenants. Worse, the tracker has no per-tenant cap, so an unbounded run causes daemon OOM.

3. **`RetryingProvider` retries forever, no circuit breaker.** When an upstream provider has a sustained outage, the existing retry policy paddles into the wall — fixed exponential backoff, no per-(tenant, provider) state, no half-open recovery probe. Production deployments need a circuit that trips after consecutive failures, opens for a backoff window, and probes for recovery.

The three are independent failure modes but share an architectural theme: **per-tenant state with bounded growth**. They compose naturally over `TenantScope` (the load-bearing type from B4) and the existing `RetryingProvider` infrastructure.

## Goals

1. **Kafka and HTTP retries are safe.** When `DaemonCommand::SubmitTask` carries an `Idempotency-Key`, a redelivery returns the existing task id — no second execution, no double-charge, no duplicate side effects. 24-hour key TTL matches the Stripe API contract.
2. **Memory-isolate per tenant.** Replace the global token tracker with `HashMap<tenant_id, TenantTokenState>`. Each tenant has a configurable in-flight cap (default 1M tokens). Submissions exceeding the cap are rejected with a structured error so callers can back off cleanly.
3. **Circuit breaker on (tenant, provider).** When 5 consecutive failures land for a `(tenant_id, provider)` pair, the circuit opens for 30s. Subsequent failures double the open window up to 5min. After the window, a single probe transitions through `HalfOpen` — success closes, failure re-opens with extended backoff.
4. **No regressions to single-tenant deployments.** All three components have defaults that preserve today's behavior. Single-tenant mode (`TenantScope::default()`) gets effectively-infinite caps and shares state in a single bucket.
5. **Composable with existing infrastructure.** Idempotency reuses the existing `tasks` table + `TaskStore` trait. The token tracker layers on top of the existing per-task token accounting. The circuit breaker wraps `LlmProvider` like `RetryingProvider` already does.

## Non-Goals

- **Per-tenant rate limiting / billing budgets.** Cost-control is a different concern from memory-isolation. Useful — but adds a per-tenant config surface (windows, quotas, refill policies) that bloats this round. Future round.
- **Per-(tenant, provider, model) circuit granularity.** Per-(tenant, provider) is sufficient for B5b. Per-model is a finer granularity that adds state without clear production value.
- **Distributed dedup across daemon instances.** Single-daemon deployments are the target; horizontal scaling shares a Postgres `tasks` table where the unique index enforces dedup naturally. Multi-region cross-daemon dedup is a separate concern (would need consensus or a Redis layer).
- **Refactoring the existing `RetryingProvider`.** The new `CircuitBreakerProvider` wraps it; existing call sites continue to use `RetryingProvider` directly. No call-site changes for clients that don't opt in to circuit breaking.
- **Migration of `heartbit-cloud`.** Cloud's adoption of the new opt-in features follows on its own cadence.

## Design

### Architecture

Three independent components landing as three sequential commits + verification + docs commit. Each composes via `TenantScope` (B4's load-bearing type).

```
┌──────────────────────────────────────────────────────────────────┐
│ Component 1: Idempotency keys on SubmitTask                      │
│   DaemonCommand::SubmitTask gains idempotency_key: Option<String>│
│   daemon_tasks: tenant_id NOT NULL DEFAULT '' (B4 pattern) +     │
│                 idempotency_key TEXT + partial UNIQUE INDEX on   │
│                 (tenant_id, idempotency_key) WHERE NOT NULL      │
│   Both InMemoryTaskStore + PostgresTaskStore implement lookup    │
│   Background sweep: NULL out keys older than 24h                 │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Component 2: Per-tenant context-overflow accounting              │
│   TenantTokenTracker { states: HashMap<TenantId, State>, cap }   │
│   Submission gate: estimated_tokens + in_flight > cap → reject   │
│   Per-turn accounting: in_flight updated on each LLM response    │
│   New error: Error::TenantOverloaded { tenant_id, in_flight, cap}│
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Component 3: Per-(tenant, provider) circuit breaker              │
│   CircuitTracker { circuits: HashMap<(Tenant, Provider), Arc<C>>}│
│   ProviderCircuit { state: Mutex<Closed | Open | HalfOpen> }     │
│   CircuitBreakerProvider<P> wraps any LlmProvider                │
│   Composes with RetryingProvider (Circuit outer, Retry inner)    │
└──────────────────────────────────────────────────────────────────┘
```

### Component 1: Idempotency keys on `DaemonCommand::SubmitTask`

**Target table.** The daemon stores tasks in `daemon_tasks` (managed by `PostgresTaskStore` in `crates/heartbit/src/daemon/store.rs`), **not** the older `tasks` table in `crates/heartbit/src/store/postgres.rs`. `daemon_tasks` already has a nullable `tenant_id TEXT` column from B4. Component 1 only needs to:

```sql
-- Phase A: tighten tenant_id to NOT NULL DEFAULT '' (matches the B4 pattern
-- already applied to audit_log). Backfills NULL rows to '' before the constraint flip.
UPDATE daemon_tasks SET tenant_id = '' WHERE tenant_id IS NULL;
ALTER TABLE daemon_tasks ALTER COLUMN tenant_id SET DEFAULT '';
ALTER TABLE daemon_tasks ALTER COLUMN tenant_id SET NOT NULL;

-- Phase B: add idempotency_key column + partial unique index
ALTER TABLE daemon_tasks ADD COLUMN IF NOT EXISTS idempotency_key TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_daemon_tasks_idem
  ON daemon_tasks (tenant_id, idempotency_key)
  WHERE idempotency_key IS NOT NULL;
```

The empty-string sentinel matches `TenantScope::default()` so single-tenant deployments remain transparent. The partial index allows multiple `NULL` values (most submissions don't supply a key) while enforcing uniqueness on `(tenant_id, key)` pairs.

**`TaskStore` trait extension.** The existing `TaskStore` trait (`crates/heartbit/src/daemon/store.rs:14`) already operates on `DaemonTask` which carries `tenant_id: Option<String>`. New methods take `&TenantScope` (or `tenant_id: &str`) and the optional idempotency key:

- `find_by_idempotency_key(tenant_id: &str, key: &str) -> Result<Option<DaemonTask>, Error>`
- `insert_with_idem(task: DaemonTask) -> Result<(), Error>` — `task.idempotency_key` is set; insert may fail with `Error::Store` on unique violation.
- `sweep_expired_idempotency_keys(cutoff: DateTime<Utc>) -> Result<usize, Error>`

**Existing submission entry points.** Both `DaemonHandle::submit_task` and `DaemonHandle::submit_task_with_user` (in `crates/heartbit/src/daemon/core.rs`) construct a `DaemonTask`, call `self.store.insert(daemon_task)`, then publish `DaemonCommand::SubmitTask` to Kafka (or HTTP-only direct dispatch). They become `submit_task_with_idem(...)` aware: when an idempotency key is provided, the lookup-then-insert-or-fallback pattern below applies before the Kafka publish.

**Wire format change.** `DaemonCommand::SubmitTask` (in `crates/heartbit/src/daemon/types.rs`) gains:

```rust
pub enum DaemonCommand {
    SubmitTask {
        // ... existing fields ...
        #[serde(default, skip_serializing_if = "Option::is_none")]
        idempotency_key: Option<String>,
    },
    // ... other variants ...
}
```

`#[serde(default)]` so existing serialized payloads still deserialize as `None`. Pre-release additive change.

**HTTP API.** The daemon's `POST /v1/tasks` (or equivalent) accepts:
- HTTP header `Idempotency-Key: <opaque-string>` (industry standard)
- OR a body field `idempotency_key`

Both populate the same field on the underlying `SubmitTask` command.

**Storage.** `DaemonTask` struct gains a new `idempotency_key: Option<String>` field (with `#[serde(default, skip_serializing_if = "Option::is_none")]`). The schema migration (above) provides the `idempotency_key TEXT` column and the `idx_daemon_tasks_idem` partial unique index.

**Submission flow.** Inside `DaemonHandle::submit_task_with_user` (the B4 entry point — see `crates/heartbit/src/daemon/core.rs:118-165`):

```rust
if let Some(ref key) = idempotency_key {
    // Look up first
    if let Some(existing_task) = self.store.find_by_idempotency_key(&user_context.tenant_id, key)? {
        return Ok(existing_task.id); // No second execution; no Kafka publish.
    }
    // Create with key. The partial unique index makes the insert atomic;
    // a concurrent insert of the same (tenant, key) fails the second insert.
    let mut daemon_task = DaemonTask::new_with_user(id, &task_str, &source_str,
        &user_context.user_id, &user_context.tenant_id);
    daemon_task.idempotency_key = Some(key.clone());
    match self.store.insert(daemon_task) {
        Ok(()) => {}
        Err(e) if is_unique_violation(&e) => {
            // Concurrent inserter won; look up and return their task id.
            return Ok(self.store
                .find_by_idempotency_key(&user_context.tenant_id, key)?
                .ok_or_else(|| Error::Daemon("unique violation but row not found".into()))?
                .id);
        }
        Err(e) => return Err(e),
    }
}
// Then continue with the normal Kafka publish.
```

The `is_unique_violation` helper (added in `crates/heartbit/src/daemon/store.rs`) inspects the error message for the Postgres unique-constraint signature `code: "23505"` — standard pattern for sqlx errors that have already been mapped to `Error::Daemon` / `Error::Store`.

**`TaskStore` trait extension.** Add to the existing trait at `crates/heartbit/src/daemon/store.rs:14`:

```rust
/// Find a task by its (tenant_id, idempotency_key) pair. Returns `None` when no
/// matching live row exists, OR when the row exists but its key has been
/// nulled out by the TTL sweep.
fn find_by_idempotency_key(
    &self,
    tenant_id: &str,
    idempotency_key: &str,
) -> Result<Option<DaemonTask>, Error>;

/// Sweep idempotency keys older than `cutoff` by setting them to NULL.
/// (We retain the rows themselves so callers that look up by `id` still hit.)
/// Returns the number of rows updated.
fn sweep_expired_idempotency_keys(
    &self,
    cutoff: chrono::DateTime<chrono::Utc>,
) -> Result<usize, Error>;
```

`insert` itself is reused — the `idempotency_key` field travels on `DaemonTask`. Both `InMemoryTaskStore` and `PostgresTaskStore` implement the new methods.

**TTL sweep.** A background task in `DaemonCore::run` (matching the existing audit-prune pattern from B4 Task 7):

```rust
if let Some(ttl) = self.config.idempotency.ttl_hours {
    let store = self.store.clone();
    let cancel = self.cancel.clone();
    let interval = std::time::Duration::from_secs(
        self.config.idempotency.sweep_interval_minutes.unwrap_or(60) * 60,
    );
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        tick.tick().await;
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = tick.tick() => {
                    let cutoff = chrono::Utc::now() - chrono::Duration::hours(ttl as i64);
                    if let Err(e) = store.sweep_expired_idempotency_keys(cutoff).await {
                        tracing::warn!(error = %e, "idempotency sweep failed");
                    }
                }
            }
        }
    });
}
```

`InMemoryTaskStore::sweep_expired_idempotency_keys` walks its `HashMap<Uuid, DaemonTask>` and clears `idempotency_key` on entries where `created_at < cutoff`. `PostgresTaskStore` runs `UPDATE daemon_tasks SET idempotency_key = NULL WHERE idempotency_key IS NOT NULL AND created_at < $1`.

**Configuration:**

```toml
[daemon.idempotency]
ttl_hours = 24                  # Default 24h; matches Stripe.
sweep_interval_minutes = 60     # How often the sweep runs.
```

Both `Option<u32>` / `Option<u64>`. Validation: zero values rejected at config-load time (matches the B4 pattern for `prune_interval_minutes`).

### Component 2: Per-tenant context-overflow accounting

**New type** in `crates/heartbit-core/src/agent/`:

```rust
pub struct TenantTokenTracker {
    states: std::sync::RwLock<HashMap<String, TenantTokenState>>,
    per_tenant_cap: usize,
}

#[derive(Debug, Default, Clone)]
pub struct TenantTokenState {
    pub in_flight: usize,
    pub high_water: usize,    // observability — peak in-flight per tenant
}

impl TenantTokenTracker {
    pub fn new(per_tenant_cap: usize) -> Self { ... }

    /// Reserve `tokens` for the given tenant. Returns Err(TenantOverloaded)
    /// if the tenant's in_flight + tokens would exceed the cap.
    ///
    /// Takes `self: &Arc<Self>` so the returned RAII guard can outlive any
    /// borrow of the tracker — required because reservations span `.await`
    /// points (LLM calls, daemon dispatch) and cross task boundaries.
    pub fn reserve(self: &Arc<Self>, scope: &TenantScope, tokens: usize) -> Result<TokenReservation, Error> {
        let tenant = scope.tenant_id.clone();
        let mut guard = self.states.write().map_err(|_| Error::Agent("token tracker poisoned".into()))?;
        let state = guard.entry(tenant.clone()).or_default();
        if state.in_flight.saturating_add(tokens) > self.per_tenant_cap {
            return Err(Error::TenantOverloaded {
                tenant_id: tenant,
                in_flight: state.in_flight,
                cap: self.per_tenant_cap,
            });
        }
        state.in_flight += tokens;
        if state.in_flight > state.high_water {
            state.high_water = state.in_flight;
        }
        Ok(TokenReservation {
            tracker: Arc::clone(self),
            tenant_id: tenant,
            tokens,
        })
    }

    /// Adjust an in-flight reservation by `delta` tokens for the given tenant.
    /// Used per-turn after each LLM response to correct the original estimate.
    /// Negative deltas decrement; positive deltas increment but never push
    /// `in_flight` above `per_tenant_cap` (excess is silently clamped — the
    /// reservation already passed the cap check at submission, and we don't
    /// want a single turn to retroactively reject the whole task).
    pub fn adjust(&self, scope: &TenantScope, delta: i64) {
        if let Ok(mut guard) = self.states.write() {
            if let Some(state) = guard.get_mut(&scope.tenant_id) {
                if delta >= 0 {
                    state.in_flight = state
                        .in_flight
                        .saturating_add(delta as usize)
                        .min(self.per_tenant_cap);
                } else {
                    state.in_flight = state.in_flight.saturating_sub((-delta) as usize);
                }
                if state.in_flight > state.high_water {
                    state.high_water = state.in_flight;
                }
            }
        }
    }

    /// Drop a reservation (e.g., when a task completes or is cancelled).
    fn release(&self, tenant_id: &str, tokens: usize) {
        if let Ok(mut guard) = self.states.write() {
            if let Some(state) = guard.get_mut(tenant_id) {
                state.in_flight = state.in_flight.saturating_sub(tokens);
            }
        }
    }

    /// Observability snapshot for metrics.
    pub fn snapshot(&self) -> Vec<(String, TenantTokenState)> { ... }
}

/// RAII reservation that releases its tokens on `Drop`. Owns an `Arc` clone
/// of the tracker (rather than a `&'a` borrow) so the reservation can be moved
/// across `.await` points and held inside futures spawned via `tokio::spawn`.
/// Matches the existing project pattern for `Arc<dyn Memory>` and similar
/// shared-state owners.
pub struct TokenReservation {
    tracker: Arc<TenantTokenTracker>,
    tenant_id: String,
    tokens: usize,
}

impl Drop for TokenReservation {
    fn drop(&mut self) {
        self.tracker.release(&self.tenant_id, self.tokens);
    }
}
```

`RwLock` on `std::sync` (not tokio) — locks never held across `.await`. Per the project's existing convention.

**New error variant** in `Error`:

```rust
#[error("tenant {tenant_id} overloaded: in_flight={in_flight}, cap={cap}")]
TenantOverloaded {
    tenant_id: String,
    in_flight: usize,
    cap: usize,
},
```

**Integration points:**

1. **Submission gate (daemon).** `DaemonCore::dispatch_command` is the single integration point — it owns the tenant scope and runs before any agent work begins. On `SubmitTask`:
   - Compute `estimated = task_input.len() / 4 + 4096`.
   - Call `tracker.reserve(&scope, estimated)` — this is the *only* `reserve()` call site.
   - On `Ok(reservation)`: stash the reservation inside the spawned task future (so its `Drop` fires when the task ends, naturally or via cancellation), then proceed with the normal dispatch path.
   - On `Err(Error::TenantOverloaded { .. })`:
     - **Kafka path:** NACK the message (do not commit the offset). Kafka will redeliver after the configured retry backoff, by which time the tenant's in-flight has likely drained. Log at `warn` with `tenant_id` and current `in_flight` for observability. **Risk:** a tenant pinned at the cap will burn redelivery cycles. Mitigation: bound redelivery attempts via Kafka consumer config; persistent overload is a billing/quota signal, not a transport-layer concern.
     - **HTTP path:** the `POST /v1/tasks` handler returns `503 Service Unavailable` with a `Retry-After: 5` header and a JSON body `{"error": "tenant_overloaded", "in_flight": <n>, "cap": <m>}`. Clients are expected to back off and retry.
     - In both cases, **no row is written to the `tasks` table** — the rejection is upstream of task creation.
   - The `AgentRunner` itself does **not** call `reserve()`; the daemon owns that decision. Standalone (non-daemon) deployments that build an `AgentRunner` directly are not subject to per-tenant overload checks (they have no tenant boundary anyway).
2. **Per-turn accounting.** Inside `AgentRunner::run`, after each LLM response, the runner calls:
   ```rust
   let actual_used = response.usage.input_tokens + response.usage.output_tokens;
   let delta = (actual_used as i64) - (reservation_so_far as i64);
   tracker.adjust(&scope, delta);
   reservation_so_far = actual_used;
   ```
   Where `reservation_so_far` is the cumulative actual usage credited to the tenant so far during this run (initialized to the original `estimated` value). The signed `delta` keeps the per-tenant `in_flight` counter accurate without double-counting. The original `TokenReservation` continues to hold its initial `estimated` tokens; on `Drop` it releases that amount, and the cumulative `adjust(..)` calls have already reconciled the difference.
3. **Release on completion.** Task completion, error, or cancellation drops the `TokenReservation`. The `Drop` impl releases the original `estimated` reservation; combined with the per-turn `adjust(..)` reconciliation, the tenant's `in_flight` returns to the pre-task baseline.

**Configuration:**

```toml
[orchestrator]
max_tokens_in_flight_per_tenant = 1000000   # 1M tokens per tenant default
```

`Option<usize>` with `#[serde(default)]`. When unset, the tracker uses an effectively-infinite cap (`usize::MAX / 2` to avoid overflow in `saturating_add`). Single-tenant deployments don't need to configure this.

**Integration with the runner.** `AgentRunnerBuilder` gains:

```rust
pub fn tenant_tracker(mut self, tracker: Arc<TenantTokenTracker>) -> Self {
    self.tenant_tracker = Some(tracker);
    self
}
```

When set, `run_loop` calls `tracker.reserve(&scope, estimated)` at start and adjusts on each turn. When unset, no tenant tracking — backward compatible.

### Component 3: Per-(tenant, provider) circuit breaker

**New types** in `crates/heartbit-core/src/llm/`:

```rust
pub struct CircuitTracker {
    circuits: std::sync::RwLock<HashMap<CircuitKey, Arc<ProviderCircuit>>>,
    config: CircuitConfig,
}

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct CircuitKey {
    pub tenant_id: String,
    pub provider: String,
}

#[derive(Debug, Clone)]
pub struct CircuitConfig {
    pub failure_threshold: u32,
    pub initial_open_duration: Duration,
    pub max_open_duration: Duration,
    pub backoff_multiplier: f64,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            initial_open_duration: Duration::from_secs(30),
            max_open_duration: Duration::from_secs(300),
            backoff_multiplier: 2.0,
        }
    }
}

pub struct ProviderCircuit {
    // parking_lot::Mutex (not std::sync::Mutex): a fault-tolerance layer that
    // disables itself permanently on a single panic defeats its purpose. The
    // poisoning-free guard means a panicked thread leaves the circuit usable
    // for the rest of the daemon's lifetime. parking_lot is already a
    // transitive dependency via tokio's internals, so this isn't a new tree.
    state: parking_lot::Mutex<CircuitState>,
    config: CircuitConfig,
}

enum CircuitState {
    Closed { consecutive_failures: u32 },
    Open { until: Instant, prev_duration: Duration },
    HalfOpen,
}

impl ProviderCircuit {
    /// Returns Err(CircuitOpen) if the circuit is currently open.
    /// Otherwise, transitions HalfOpen → "single probe in flight" or stays Closed.
    /// Returns a `CircuitPermit` owning an `Arc` clone of `self`, so the permit
    /// can be moved across `.await` points and into spawned tasks.
    pub fn permit(self: &Arc<Self>) -> Result<CircuitPermit, Error> {
        let mut state = self.state.lock();  // parking_lot: no Result, no poisoning
        match *state {
            CircuitState::Closed { .. } => Ok(CircuitPermit { circuit: Arc::clone(self) }),
            CircuitState::Open { until, prev_duration } => {
                if Instant::now() >= until {
                    *state = CircuitState::HalfOpen;
                    Ok(CircuitPermit { circuit: Arc::clone(self) })
                } else {
                    Err(Error::CircuitOpen { until, prev_duration })
                }
            }
            CircuitState::HalfOpen => Err(Error::CircuitOpen {
                until: Instant::now() + Duration::from_millis(50),
                prev_duration: Duration::ZERO,
            }),
        }
    }

    fn record_success(&self) {
        let mut state = self.state.lock();
        *state = CircuitState::Closed { consecutive_failures: 0 };
    }

    fn record_failure(&self) {
        let mut state = self.state.lock();
        match *state {
            CircuitState::Closed { consecutive_failures } => {
                let n = consecutive_failures + 1;
                if n >= self.config.failure_threshold {
                    *state = CircuitState::Open {
                        until: Instant::now() + self.config.initial_open_duration,
                        prev_duration: self.config.initial_open_duration,
                    };
                } else {
                    *state = CircuitState::Closed { consecutive_failures: n };
                }
            }
            CircuitState::HalfOpen => {
                let new_duration = std::cmp::min(
                    Duration::from_secs_f64(self.config.backoff_multiplier),
                    self.config.max_open_duration,
                );
                *state = CircuitState::Open {
                    until: Instant::now() + new_duration,
                    prev_duration: new_duration,
                };
            }
            CircuitState::Open { .. } => { /* already open; no-op */ }
        }
    }
}

/// `Arc`-owning permit so it can outlive any borrow of the circuit and survive
/// movement across `.await`. Same rationale as `TokenReservation` (Component 2).
pub struct CircuitPermit {
    circuit: Arc<ProviderCircuit>,
}

impl CircuitPermit {
    pub fn record_success(self) { self.circuit.record_success(); }
    pub fn record_failure(self) { self.circuit.record_failure(); }
}
```

**New error variants:**

```rust
#[error("circuit breaker open: retry after {until:?}")]
CircuitOpen {
    until: Instant,
    prev_duration: Duration,
},
```

**Wrapper provider.** `CompletionRequest` does not carry tenant identity. Rather than bolting metadata onto requests, the wrapper takes the `TenantScope` at construction — each `AgentRunner` already has tenant identity (`audit_tenant_id` / `audit_user_id`) and is the natural place to build a per-runner wrapper that points at the *shared* `Arc<CircuitTracker>`:

```rust
pub struct CircuitBreakerProvider<P: LlmProvider> {
    inner: P,
    tracker: Arc<CircuitTracker>,
    provider_name: String,
    scope: TenantScope,
}

impl<P: LlmProvider> CircuitBreakerProvider<P> {
    pub fn new(
        inner: P,
        tracker: Arc<CircuitTracker>,
        provider_name: impl Into<String>,
        scope: TenantScope,
    ) -> Self {
        Self {
            inner,
            tracker,
            provider_name: provider_name.into(),
            scope,
        }
    }
}

impl<P: LlmProvider> LlmProvider for CircuitBreakerProvider<P> {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
        let circuit = self.tracker.circuit_for(&self.scope, &self.provider_name);
        let permit = circuit.permit()?;
        let result = self.inner.complete(request).await;
        match &result {
            Ok(_) => permit.record_success(),
            Err(e) if is_circuit_failure(e) => permit.record_failure(),
            Err(_) => permit.record_success(), // Non-circuit-tripping errors don't count
        }
        result
    }

    async fn stream_complete(
        &self,
        request: CompletionRequest,
        on_text: &super::OnText,
    ) -> Result<CompletionResponse, Error> {
        let circuit = self.tracker.circuit_for(&self.scope, &self.provider_name);
        let permit = circuit.permit()?;
        let result = self.inner.stream_complete(request, on_text).await;
        match &result {
            Ok(_) => permit.record_success(),
            Err(e) if is_circuit_failure(e) => permit.record_failure(),
            Err(_) => permit.record_success(),
        }
        result
    }

    fn model_name(&self) -> Option<&str> {
        self.inner.model_name()
    }
}
```

`CircuitTracker::circuit_for(&scope, provider)` returns an existing `Arc<ProviderCircuit>` from the inner `HashMap`, or inserts a new one with `CircuitConfig::default()` and returns the clone. The Arc-owning permit (defined above) is returned by `permit()`.

**Error classification.** The `is_circuit_failure` predicate uses the existing `crate::llm::error_class::classify()` from heartbit-core:

```rust
fn is_circuit_failure(err: &Error) -> bool {
    use crate::llm::error_class::ErrorClass;
    matches!(
        crate::llm::error_class::classify(err),
        ErrorClass::ServerError | ErrorClass::RateLimited | ErrorClass::Network
    )
    // ContextOverflow handled by auto-compaction, not circuit
    // AuthError won't recover from circuit; needs manual fix
    // InvalidRequest is a 4xx that won't change on retry
}
```

**Configuration:**

```toml
[provider.circuit]
failure_threshold = 5
initial_open_duration_seconds = 30
max_open_duration_seconds = 300
backoff_multiplier = 2.0
```

All optional with sensible defaults.

**Integration with `RetryingProvider`.** Composition order matters and is fixed by this design:

```
CircuitBreakerProvider<RetryingProvider<AnthropicProvider>>
   (outer)               (inner)
```

**Permit accounting (explicit decision).** Each call to the *outer* `CircuitBreakerProvider::complete` consumes exactly **one** permit. The wrapped `RetryingProvider` handles its full retry budget (default: 3 attempts with exponential backoff) *inside* that one permit. So a circuit configured at `failure_threshold = 5` opens after **5 consecutive retry-exhausted outer attempts** — i.e., 5 sustained upstream outages — not 5 raw HTTP failures.

Why outer-circuit / inner-retry:
- Putting retry outside the circuit would let the retry layer hammer an `Open` circuit, defeating its purpose.
- Putting circuit inside retry would mean each transient failure burns a circuit failure slot, blowing the threshold open on a momentary blip — also wrong.
- The chosen order means: transient blips are absorbed by retry; sustained outages exhaust retry, count as a circuit failure, eventually open the circuit; an open circuit short-circuits without burning any retry budget.

When the circuit is open, no retries fire — the agent gets an immediate `CircuitOpen` error and can react (e.g., fall through to a different provider). This composition is already supported by the existing `LlmProvider` trait; no trait changes.

### CLI / config wiring

```toml
[daemon.idempotency]
ttl_hours = 24
sweep_interval_minutes = 60

[orchestrator]
max_tokens_in_flight_per_tenant = 1000000

[provider.circuit]
failure_threshold = 5
initial_open_duration_seconds = 30
max_open_duration_seconds = 300
backoff_multiplier = 2.0
```

Env-var overrides:

- `HEARTBIT_IDEMPOTENCY_TTL_HOURS`
- `HEARTBIT_TENANT_TOKEN_CAP`
- `HEARTBIT_CIRCUIT_FAILURE_THRESHOLD`

Validation: zero values rejected at parse time for all `Option<u32>` settings (matches the B4 pattern).

CLI builds:
- `Arc<TenantTokenTracker>` from `[orchestrator].max_tokens_in_flight_per_tenant` and threads into the `AgentRunnerBuilder`.
- `Arc<CircuitTracker>` from `[provider.circuit]` and wraps each provider with `CircuitBreakerProvider`.
- Daemon spawns the idempotency-key sweep task on startup.

### Test plan

Per the project's TDD discipline, each component lands with its own tests.

**Idempotency (~12 tests):**

- `submit_with_idem_key_returns_existing_task_on_redelivery` — InMemoryTaskStore.
- `submit_with_idem_key_under_different_tenant_creates_new_task` — tenant isolation.
- `submit_without_idem_key_creates_new_task_each_time` — backward compat.
- `concurrent_submits_same_key_resolve_to_same_task_id` — tests the unique-violation fallback path. Uses `tokio::join!` to fire 10 concurrent submits.
- `sweep_expires_keys_older_than_ttl` — verifies the TTL sweep nulls out old keys.
- `sweep_does_not_touch_recent_keys` — boundary condition.
- (Postgres integration: `#[ignore = "requires DATABASE_URL"]` versions of the above 6.)

**Tenant token tracker (~10 tests):**

- `reserve_within_cap_succeeds_and_increments_in_flight`
- `reserve_exceeding_cap_returns_tenant_overloaded`
- `release_decrements_in_flight`
- `reservation_drop_releases_tokens` — RAII via `Drop`.
- `tenant_isolation_does_not_share_in_flight` — two tenants with separate counters.
- `default_scope_uses_single_bucket` — empty-string sentinel.
- `high_water_tracks_peak_correctly`
- `snapshot_returns_per_tenant_state_for_observability`
- `concurrent_reservations_serialize_correctly_under_rwlock`
- `cap_zero_at_construction_treated_as_infinite_or_rejected` — design choice: validate at builder time.

**Circuit breaker (~15 tests):**

- `closed_circuit_passes_requests_through`
- `n_consecutive_failures_opens_circuit` — N = `failure_threshold`.
- `successful_request_resets_consecutive_failures` — partial-failure recovery.
- `non_tripping_errors_do_not_increment_failures` — auth/4xx/context-overflow.
- `open_circuit_rejects_with_circuit_open_error`
- `open_circuit_transitions_to_half_open_after_duration`
- `half_open_success_closes_circuit`
- `half_open_failure_re_opens_with_doubled_duration` — exponential backoff.
- `repeated_half_open_failures_clamp_at_max_open_duration`
- `tenant_isolation_separate_circuits_per_tenant`
- `provider_isolation_separate_circuits_per_provider`
- `concurrent_requests_during_half_open_only_one_probes` — single-probe semantics.
- `circuit_state_transitions_via_record_success/failure_compose_correctly`
- `circuit_breaker_provider_wrapper_threads_tenant_scope_correctly`
- `circuit_breaker_composes_with_retrying_provider` — outer-circuit / inner-retry pattern.

Estimated total: ~37 new unit tests + ~6 Postgres integration tests (`#[ignore]`-gated).

## Sequencing

| # | Commit | Notes |
|---|---|---|
| 1 | `feat(daemon): idempotency keys on SubmitTask` | Schema migration in `PostgresTaskStore::run_migration` (additive — tightens `daemon_tasks.tenant_id`, adds `idempotency_key` + partial unique index); `DaemonCommand::SubmitTask.idempotency_key` and `DaemonTask.idempotency_key` fields; `TaskStore::find_by_idempotency_key` + `sweep_expired_idempotency_keys` methods; both `InMemoryTaskStore` and `PostgresTaskStore` impls; HTTP API header support; daemon background sweep task. |
| 2 | `feat(core): per-tenant context-overflow tracker` | `TenantTokenTracker` + `TenantTokenState` + `TokenReservation` (RAII) types; `Error::TenantOverloaded` variant; `AgentRunnerBuilder::tenant_tracker` builder; submission-time reserve + per-turn adjustment. |
| 3 | `feat(provider): per-(tenant, provider) circuit breaker` | `CircuitTracker` + `ProviderCircuit` + `CircuitState` + `CircuitConfig` + `CircuitPermit` types; `Error::CircuitOpen` variant; `CircuitBreakerProvider<P>` wrapper; CLI wiring. |
| — | Verification matrix | Cross-feature builds + new unit tests + (optional) Postgres integration tests. |
| 4 | `docs: B5b CHANGELOG + recipe` | User-facing recipe under `book/src/recipes/failure-modes.md` covering the three components. CHANGELOG entries documenting the new types, configuration, and the pre-release additive change to `DaemonCommand::SubmitTask`. |

Each commit must keep `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace --lib` green.

## Risks

1. **Idempotency unique-index race.** Concurrent inserts of the same key — second insert hits `SQLSTATE 23505` (Postgres unique violation). Mitigated by the explicit catch-and-fallback-to-lookup pattern. Standard idempotency implementation pattern; well-understood. Tested explicitly by `concurrent_submits_same_key_resolve_to_same_task_id`.

2. **Token estimation accuracy.** Submission-time `tokens = task_input.len() / 4 + 4096` is approximate. Real costs tracked per-turn after the run starts. Worst case: a tenant slightly overshoots the cap because the estimate undercounted. Mitigated by setting the cap with headroom (1M default; production deployments configure based on observed peak `high_water` values). Future round can tighten via tokenizer-based estimation if needed.

3. **Circuit breaker thrashing.** A provider failing intermittently could oscillate `Open → HalfOpen → Open`. Mitigated by exponential backoff on subsequent opens (30s → 1m → 2m → 4m, capped at 5m) and by `max_open_duration` clamp. Behavior tested by `repeated_half_open_failures_clamp_at_max_open_duration`.

4. **Per-tenant state growth.** `HashMap<tenant_id, ...>` grows with the number of distinct tenants. Bounded in practice for known production deployments (~100s of tenants). For runaway tenant turnover (e.g., trial sign-ups), an LRU eviction policy can be added in a future round. Tracking this in observability via the `snapshot()` method.

5. **Pre-release breaking change to `DaemonCommand::SubmitTask`.** The new `idempotency_key` field is optional + `#[serde(default)]` + `#[serde(skip_serializing_if = "Option::is_none")]`. Existing serialized payloads parse fine (field is `None`); new payloads with the field don't break consumers that ignore unknown fields. Lockstep upgrade not required.

6. **Mutex poisoning.** Component 2 (`TenantTokenTracker`) uses `std::sync::RwLock` and surfaces poisoning as `Error::Agent("token tracker poisoned")` so a panicked tenant doesn't disable the daemon for everyone — the affected tenant just stops being able to reserve, which is acceptable graceful degradation. Component 3 (`ProviderCircuit`) uses **`parking_lot::Mutex`** (no poisoning) — a fault-tolerance layer that disables itself permanently on a single panic defeats its purpose. Combined with the project convention that locks are never held across `.await`, the surface area for a panic-while-locked is small (only synchronous state-machine transitions hold the lock).

7. **HalfOpen single-probe race.** Two concurrent requests hitting an `Open` circuit at the moment `until` elapses could both transition to `HalfOpen`. The mutex guards the state transition, so only one wins; the other re-reads `HalfOpen` and gets `CircuitOpen` back. Tested by `concurrent_requests_during_half_open_only_one_probes`.

## Out-of-Scope (deferred)

- **Per-tenant rate limiting / billing budgets.** Cost-control. Future round.
- **Per-(tenant, provider, model) circuit granularity.** B5b uses (tenant, provider).
- **Distributed dedup.** Single-daemon today; horizontal scaling shares the Postgres `tasks` table.
- **`heartbit-cloud` adoption.** Cloud's adoption follows on its own cadence.
- **Refactoring `RetryingProvider`.** B5b adds a wrapper, doesn't change the existing one.
