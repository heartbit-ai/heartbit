//! Per-(tenant, provider) circuit breaker state machine.
//!
//! See `docs/superpowers/specs/2026-05-02-b5b-failure-mode-hardening-design.md`
//! Component 3 for design rationale.
//!
//! Locking:
//! - `ProviderCircuit::state` uses `parking_lot::Mutex` (no poisoning). A
//!   fault-tolerance layer that disables itself permanently on a single panic
//!   defeats its purpose.
//! - `CircuitTracker::circuits` uses `std::sync::RwLock` (matches the standalone
//!   project convention for never-await locks like `InMemoryStore`/`McpSession`).
//!   Poisoning is unreachable in practice — the write-lock body only does
//!   `HashMap::entry().or_insert_with(...)` + `Arc::new(ProviderCircuit::new(...))`,
//!   neither of which can panic.

#![allow(missing_docs)]
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::auth::TenantScope;
use crate::error::Error;

/// Configuration parameters for the per-provider circuit breaker state machine.
///
/// Controls the failure threshold that trips the circuit open, the initial
/// cooldown duration, the maximum backoff cap, and the exponential multiplier
/// applied when a half-open probe fails. The defaults (`failure_threshold = 5`,
/// `initial_open_duration = 30s`, `max = 300s`, `multiplier = 2.0`) are
/// conservative starting points suitable for most production deployments.
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

#[derive(Debug)]
enum CircuitState {
    Closed {
        consecutive_failures: u32,
    },
    Open {
        until: Instant,
        prev_duration: Duration,
    },
    HalfOpen,
}

/// State machine for a single `(tenant, provider)` circuit.
///
/// Transitions through three states: `Closed` (normal operation, counting
/// consecutive failures), `Open` (blocking requests until the cooldown expires),
/// and `HalfOpen` (allowing one probe request to test recovery). Callers
/// obtain a `CircuitPermit` via `permit()` and must call `record_success` or
/// `record_failure` on it; a permit that is dropped without recording conservatively
/// counts as a failure so the half-open state never wedges. The internal lock is
/// `parking_lot::Mutex` so it never poisons on panic.
pub struct ProviderCircuit {
    state: Mutex<CircuitState>,
    config: CircuitConfig,
}

/// Arc-owning permit so it can outlive any borrow of the circuit and survive
/// movement across `.await`.
///
/// On drop, if neither `record_success` nor `record_failure` was called
/// (e.g., on panic or task cancellation), the permit conservatively records a
/// failure. This prevents the HalfOpen state from wedging indefinitely when a
/// probe is dropped mid-flight.
pub struct CircuitPermit {
    circuit: Arc<ProviderCircuit>,
    consumed: std::sync::atomic::AtomicBool,
}

impl std::fmt::Debug for CircuitPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CircuitPermit")
    }
}

impl CircuitPermit {
    pub fn record_success(self) {
        self.consumed
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.circuit.record_success();
    }
    pub fn record_failure(self) {
        self.consumed
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.circuit.record_failure();
    }
}

impl Drop for CircuitPermit {
    fn drop(&mut self) {
        // Conservative: if neither record_success nor record_failure was called
        // (e.g., panic, cancellation, or wrapper bug), treat the permit as a
        // failed probe. This prevents the half-open state from wedging forever.
        if !self.consumed.load(std::sync::atomic::Ordering::SeqCst) {
            self.circuit.record_failure();
        }
    }
}

impl ProviderCircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            state: Mutex::new(CircuitState::Closed {
                consecutive_failures: 0,
            }),
            config,
        }
    }

    /// Returns `Err(CircuitOpen)` if the circuit is currently open.
    /// Otherwise transitions HalfOpen → "single probe in flight" or stays Closed.
    pub fn permit(self: &Arc<Self>) -> Result<CircuitPermit, Error> {
        let mut state = self.state.lock();
        match *state {
            CircuitState::Closed { .. } => Ok(CircuitPermit {
                circuit: Arc::clone(self),
                consumed: std::sync::atomic::AtomicBool::new(false),
            }),
            CircuitState::Open {
                until,
                prev_duration,
            } => {
                if Instant::now() >= until {
                    *state = CircuitState::HalfOpen;
                    Ok(CircuitPermit {
                        circuit: Arc::clone(self),
                        consumed: std::sync::atomic::AtomicBool::new(false),
                    })
                } else {
                    Err(Error::CircuitOpen {
                        until,
                        prev_duration,
                    })
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
        *state = CircuitState::Closed {
            consecutive_failures: 0,
        };
    }

    fn record_failure(&self) {
        let mut state = self.state.lock();
        match *state {
            CircuitState::Closed {
                consecutive_failures,
            } => {
                let n = consecutive_failures + 1;
                *state = if n >= self.config.failure_threshold {
                    CircuitState::Open {
                        until: Instant::now() + self.config.initial_open_duration,
                        prev_duration: self.config.initial_open_duration,
                    }
                } else {
                    CircuitState::Closed {
                        consecutive_failures: n,
                    }
                };
            }
            CircuitState::HalfOpen => {
                let new_dur_secs = self.config.initial_open_duration.as_secs_f64()
                    * self.config.backoff_multiplier;
                let new_dur =
                    Duration::from_secs_f64(new_dur_secs).min(self.config.max_open_duration);
                *state = CircuitState::Open {
                    until: Instant::now() + new_dur,
                    prev_duration: new_dur,
                };
            }
            CircuitState::Open { .. } => { /* already open; no-op */ }
        }
    }
}

/// Composite key for the `CircuitTracker` registry.
///
/// Identifies a unique `(tenant_id, provider)` pair. Two agent runners serving
/// different tenants against the same provider share no circuit state, so a
/// misbehaving tenant cannot trip the circuit for other tenants.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct CircuitKey {
    pub tenant_id: String,
    pub provider: String,
}

/// Registry that owns one [`ProviderCircuit`] per `(tenant_id, provider)` pair.
///
/// Locking strategy: `std::sync::RwLock` (never held across `.await`).
/// Fast path: read lock + `Arc::clone`. Slow path (first insert): write lock
/// with double-check to avoid duplicate allocation under races.
pub struct CircuitTracker {
    circuits: RwLock<HashMap<CircuitKey, Arc<ProviderCircuit>>>,
    config: CircuitConfig,
}

impl CircuitTracker {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            circuits: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Get or create the circuit for `(scope.tenant_id, provider)`.
    pub fn circuit_for(&self, scope: &TenantScope, provider: &str) -> Arc<ProviderCircuit> {
        let key = CircuitKey {
            tenant_id: scope.tenant_id.clone(),
            provider: provider.to_string(),
        };
        // Fast path: read lock + clone if present.
        if let Ok(g) = self.circuits.read()
            && let Some(c) = g.get(&key)
        {
            return Arc::clone(c);
        }
        // Slow path: write lock with double-check.
        let mut g = self.circuits.write().expect("circuit tracker poisoned");
        Arc::clone(
            g.entry(key)
                .or_insert_with(|| Arc::new(ProviderCircuit::new(self.config.clone()))),
        )
    }
}

/// Classify whether an error from a provider call should count as a
/// circuit-tripping failure.
///
/// Trips: `ServerError` (500/502/503/529), `RateLimited` (429),
/// `Network` (TCP/DNS/TLS/timeout — transport failure).
/// Does NOT trip: `AuthError` (401/403 — permanent, won't recover by waiting),
/// `InvalidRequest` (400 — caller bug), `ContextOverflow` (handled by
/// auto-compaction, not the circuit), `Unknown`.
pub fn is_circuit_failure(err: &Error) -> bool {
    use crate::llm::error_class::ErrorClass;
    matches!(
        crate::llm::error_class::classify(err),
        ErrorClass::ServerError | ErrorClass::RateLimited | ErrorClass::Network
    )
}

/// Per-runner wrapper around any [`super::LlmProvider`]. Looks up the
/// per-(tenant, provider) circuit from the shared [`Arc<CircuitTracker>`] and
/// gates each request on a permit.
///
/// ## Composition order
///
/// The intended layering is `CircuitBreakerProvider<RetryingProvider<P>>`.
/// One `CircuitPermit` covers the full retry budget of the inner
/// `RetryingProvider`: the circuit counts a failure only when all retries are
/// exhausted, not on individual transient blips. Set `failure_threshold = 5`
/// to mean "5 retry-exhausted outer attempts before opening".
///
/// ## Tenant identity
///
/// `CompletionRequest` does not carry tenant scope; each `AgentRunner` binds
/// the scope at construction time via [`CircuitBreakerProvider::new`].
///
/// ## Backoff note
///
/// When a HalfOpen probe fails, the reopened duration is
/// `initial_open_duration * backoff_multiplier`, NOT `prev_duration *
/// backoff_multiplier`. This means the backoff is constant-after-first-failure
/// rather than compounding.
///
/// ## Non-circuit-tripping errors
///
/// Auth errors (401/403), bad requests (400), and context-overflow errors are
/// NOT counted as circuit failures. From the circuit's perspective these are
/// caller-side issues and the provider is healthy — such errors are recorded as
/// `record_success` so they don't accumulate toward the threshold.
pub struct CircuitBreakerProvider<P: super::LlmProvider> {
    inner: P,
    tracker: Arc<CircuitTracker>,
    provider_name: String,
    scope: TenantScope,
}

impl<P: super::LlmProvider> CircuitBreakerProvider<P> {
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

impl<P: super::LlmProvider> super::LlmProvider for CircuitBreakerProvider<P> {
    fn model_name(&self) -> Option<&str> {
        self.inner.model_name()
    }

    async fn complete(
        &self,
        request: super::types::CompletionRequest,
    ) -> Result<super::types::CompletionResponse, Error> {
        let circuit = self.tracker.circuit_for(&self.scope, &self.provider_name);
        let permit = circuit.permit()?;
        let result = self.inner.complete(request).await;
        match &result {
            Ok(_) => permit.record_success(),
            // Circuit-tripping error (5xx, 429, network): provider is degraded.
            Err(e) if is_circuit_failure(e) => permit.record_failure(),
            // Non-circuit-tripping error (auth, 4xx, etc.): provider is healthy
            // from the circuit's POV; record success so caller bugs / permanent
            // auth failures don't accumulate toward the threshold.
            Err(_) => permit.record_success(),
        }
        result
    }

    async fn stream_complete(
        &self,
        request: super::types::CompletionRequest,
        on_text: &super::OnText,
    ) -> Result<super::types::CompletionResponse, Error> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CircuitConfig {
        CircuitConfig {
            failure_threshold: 3,
            initial_open_duration: Duration::from_millis(50),
            max_open_duration: Duration::from_millis(500),
            backoff_multiplier: 2.0,
        }
    }

    #[test]
    fn closed_circuit_passes_requests() {
        let c = Arc::new(ProviderCircuit::new(cfg()));
        let p = c.permit().unwrap();
        p.record_success();
    }

    #[test]
    fn n_failures_open_circuit() {
        let c = Arc::new(ProviderCircuit::new(cfg()));
        for _ in 0..3 {
            let p = c.permit().unwrap();
            p.record_failure();
        }
        let err = c.permit().unwrap_err();
        assert!(matches!(err, Error::CircuitOpen { .. }));
    }

    #[test]
    fn success_resets_consecutive_failures() {
        let c = Arc::new(ProviderCircuit::new(cfg()));
        c.permit().unwrap().record_failure();
        c.permit().unwrap().record_failure();
        c.permit().unwrap().record_success();
        // Still under threshold after one more failure
        c.permit().unwrap().record_failure();
        assert!(c.permit().is_ok());
    }

    #[test]
    fn open_transitions_to_half_open_after_duration() {
        let c = Arc::new(ProviderCircuit::new(cfg()));
        for _ in 0..3 {
            c.permit().unwrap().record_failure();
        }
        std::thread::sleep(Duration::from_millis(60));
        assert!(c.permit().is_ok(), "should be HalfOpen permit");
    }

    #[test]
    fn half_open_success_closes_circuit() {
        let c = Arc::new(ProviderCircuit::new(cfg()));
        for _ in 0..3 {
            c.permit().unwrap().record_failure();
        }
        std::thread::sleep(Duration::from_millis(60));
        c.permit().unwrap().record_success();
        // Closed now: many permits in a row — consume each one so the Drop
        // impl doesn't count unrecorded drops as failures.
        for _ in 0..10 {
            let p = c.permit();
            assert!(p.is_ok());
            p.unwrap().record_success();
        }
    }

    #[test]
    fn half_open_failure_reopens_with_doubled_duration() {
        let c = Arc::new(ProviderCircuit::new(cfg()));
        for _ in 0..3 {
            c.permit().unwrap().record_failure();
        }
        // Wait for initial_open_duration (50ms) to expire → enters HalfOpen.
        std::thread::sleep(Duration::from_millis(70));
        // Probe in HalfOpen fails → reopens with doubled duration: 100ms.
        c.permit().unwrap().record_failure();
        // After 60ms the 100ms window has not expired yet → still open.
        std::thread::sleep(Duration::from_millis(60));
        assert!(c.permit().is_err());
        // After another 60ms (120ms total) → window has expired → HalfOpen available.
        std::thread::sleep(Duration::from_millis(60));
        assert!(c.permit().is_ok());
    }

    #[test]
    fn repeated_half_open_failures_clamp_at_max() {
        let c = Arc::new(ProviderCircuit::new(CircuitConfig {
            failure_threshold: 1,
            initial_open_duration: Duration::from_millis(100),
            max_open_duration: Duration::from_millis(150),
            backoff_multiplier: 4.0,
        }));
        c.permit().unwrap().record_failure(); // → Open(100ms)
        std::thread::sleep(Duration::from_millis(110));
        c.permit().unwrap().record_failure(); // → Open(min(400, 150) = 150ms)
        std::thread::sleep(Duration::from_millis(160));
        assert!(
            c.permit().is_ok(),
            "should be openable again at clamped duration"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn permit_can_be_moved_across_await() {
        // Compile-time check for Arc-ownership.
        let c = Arc::new(ProviderCircuit::new(cfg()));
        let p = c.permit().unwrap();
        let task = tokio::spawn(async move {
            tokio::task::yield_now().await;
            p.record_success();
        });
        task.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_requests_during_half_open_only_one_probes() {
        // Once Open transitions to HalfOpen, only one permit is granted at a time.
        // Subsequent permit attempts get CircuitOpen until the probe resolves.
        let c = Arc::new(ProviderCircuit::new(CircuitConfig {
            failure_threshold: 1,
            initial_open_duration: Duration::from_millis(20),
            max_open_duration: Duration::from_millis(200),
            backoff_multiplier: 2.0,
        }));
        c.permit().unwrap().record_failure(); // Open
        tokio::time::sleep(Duration::from_millis(30)).await;

        // First permit transitions Open → HalfOpen and is granted.
        let probe = c.permit().expect("first probe granted");

        // Second concurrent attempt while HalfOpen: rejected with CircuitOpen.
        let second = c.permit();
        assert!(matches!(second, Err(Error::CircuitOpen { .. })));

        // Probe records success → Closed
        probe.record_success();
        assert!(c.permit().is_ok());
    }

    #[test]
    fn tracker_returns_same_arc_for_same_key() {
        let t = CircuitTracker::new(cfg());
        let a = t.circuit_for(&TenantScope::new("acme"), "anthropic");
        let b = t.circuit_for(&TenantScope::new("acme"), "anthropic");
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn tracker_isolates_tenants() {
        let t = CircuitTracker::new(cfg());
        let a = t.circuit_for(&TenantScope::new("acme"), "anthropic");
        let b = t.circuit_for(&TenantScope::new("globex"), "anthropic");
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn tracker_isolates_providers() {
        let t = CircuitTracker::new(cfg());
        let a = t.circuit_for(&TenantScope::new("acme"), "anthropic");
        let b = t.circuit_for(&TenantScope::new("acme"), "openai");
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn is_circuit_failure_classifies_correctly() {
        // Server error → trips
        let server = Error::Api {
            status: 503,
            message: "service unavailable".into(),
        };
        assert!(is_circuit_failure(&server));

        // Rate limited → trips
        let rate = Error::Api {
            status: 429,
            message: "too many requests".into(),
        };
        assert!(is_circuit_failure(&rate));

        // Network/transport error → trips (sustained outage must open circuit)
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        let http_err = rt
            .block_on(reqwest::get("http://[::0]:1"))
            .expect_err("should fail");
        assert!(is_circuit_failure(&Error::Http(http_err)));

        // Auth error → does NOT trip (won't recover from retry)
        let auth = Error::Api {
            status: 401,
            message: "unauthorized".into(),
        };
        assert!(!is_circuit_failure(&auth));

        // Bad request → does NOT trip
        let bad = Error::Api {
            status: 400,
            message: "bad json".into(),
        };
        assert!(!is_circuit_failure(&bad));
    }

    // --- CircuitBreakerProvider tests ---

    use crate::llm::LlmProvider;
    use crate::llm::types::{CompletionRequest, Message};

    struct FailingProvider {
        error: Box<dyn Fn() -> Error + Send + Sync>,
    }

    impl LlmProvider for FailingProvider {
        async fn complete(
            &self,
            _r: CompletionRequest,
        ) -> Result<crate::llm::types::CompletionResponse, Error> {
            Err((self.error)())
        }
    }

    fn dummy_request() -> CompletionRequest {
        CompletionRequest {
            system: "test".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 10,
            tool_choice: None,
            reasoning_effort: None,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn circuit_opens_after_threshold_failures() {
        let tracker = Arc::new(CircuitTracker::new(CircuitConfig {
            failure_threshold: 3,
            initial_open_duration: Duration::from_secs(60),
            max_open_duration: Duration::from_secs(120),
            backoff_multiplier: 2.0,
        }));
        let inner = FailingProvider {
            error: Box::new(|| Error::Api {
                status: 503,
                message: "down".into(),
            }),
        };
        let wrapper = CircuitBreakerProvider::new(
            inner,
            tracker.clone(),
            "anthropic",
            TenantScope::new("acme"),
        );

        // Three failing calls → circuit opens
        for _ in 0..3 {
            let _ = wrapper.complete(dummy_request()).await;
        }
        // Fourth call: short-circuits with CircuitOpen, no inner call
        let err = wrapper.complete(dummy_request()).await.unwrap_err();
        assert!(matches!(err, Error::CircuitOpen { .. }));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn auth_errors_do_not_trip_circuit() {
        let tracker = Arc::new(CircuitTracker::new(cfg()));
        let inner = FailingProvider {
            error: Box::new(|| Error::Api {
                status: 401,
                message: "no key".into(),
            }),
        };
        let wrapper = CircuitBreakerProvider::new(
            inner,
            tracker.clone(),
            "anthropic",
            TenantScope::new("acme"),
        );

        for _ in 0..10 {
            let _ = wrapper.complete(dummy_request()).await;
        }
        // Circuit still closed: 401s don't trip
        let circuit = tracker.circuit_for(&TenantScope::new("acme"), "anthropic");
        assert!(circuit.permit().is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn circuit_outer_retry_inner_one_permit_per_outer_call() {
        // Wrapped: CircuitBreaker<Retrying<FailingProvider>>
        let tracker = Arc::new(CircuitTracker::new(CircuitConfig {
            failure_threshold: 2,
            initial_open_duration: Duration::from_secs(60),
            max_open_duration: Duration::from_secs(120),
            backoff_multiplier: 2.0,
        }));
        let inner = FailingProvider {
            error: Box::new(|| Error::Api {
                status: 503,
                message: "down".into(),
            }),
        };
        let retrying = crate::llm::retry::RetryingProvider::new(
            inner,
            crate::llm::retry::RetryConfig {
                max_retries: 3,
                base_delay: Duration::from_millis(1),
                max_delay: Duration::from_millis(10),
            },
        );
        let wrapper = CircuitBreakerProvider::new(
            retrying,
            tracker.clone(),
            "anthropic",
            TenantScope::new("acme"),
        );

        // Two outer calls → 2 circuit failures → opens.
        let _ = wrapper.complete(dummy_request()).await;
        let _ = wrapper.complete(dummy_request()).await;
        let err = wrapper.complete(dummy_request()).await.unwrap_err();
        assert!(matches!(err, Error::CircuitOpen { .. }));
    }

    #[test]
    fn permit_drop_without_consume_records_failure() {
        // Verifies the Drop impl safety net: an unconsumed permit counts as failure.
        let c = Arc::new(ProviderCircuit::new(CircuitConfig {
            failure_threshold: 1,
            initial_open_duration: Duration::from_millis(50),
            max_open_duration: Duration::from_millis(500),
            backoff_multiplier: 2.0,
        }));
        // Get a permit without consuming it — simulates a cancelled task.
        let permit = c.permit().unwrap();
        drop(permit); // Drop without record_* → Drop impl fires record_failure
        // With threshold=1, one failure → circuit opens.
        assert!(
            c.permit().is_err(),
            "circuit should be open after unconsumed permit drop"
        );
    }
}
