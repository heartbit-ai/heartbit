//! Per-(tenant, provider) circuit breaker state machine.
//!
//! See `docs/superpowers/specs/2026-05-02-b5b-failure-mode-hardening-design.md`
//! Component 3 for design rationale.
//!
//! Locking: `parking_lot::Mutex` (no poisoning). A fault-tolerance layer
//! that disables itself permanently on a single panic defeats its purpose.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::error::Error;

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

pub struct ProviderCircuit {
    state: Mutex<CircuitState>,
    config: CircuitConfig,
}

/// Arc-owning permit so it can outlive any borrow of the circuit and survive
/// movement across `.await`.
pub struct CircuitPermit {
    circuit: Arc<ProviderCircuit>,
}

impl std::fmt::Debug for CircuitPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CircuitPermit")
    }
}

impl CircuitPermit {
    pub fn record_success(self) {
        self.circuit.record_success();
    }
    pub fn record_failure(self) {
        self.circuit.record_failure();
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
            }),
            CircuitState::Open {
                until,
                prev_duration,
            } => {
                if Instant::now() >= until {
                    *state = CircuitState::HalfOpen;
                    Ok(CircuitPermit {
                        circuit: Arc::clone(self),
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
        // Closed now: many permits in a row.
        for _ in 0..10 {
            assert!(c.permit().is_ok());
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
}
