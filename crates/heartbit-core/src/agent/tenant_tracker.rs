//! Per-tenant in-flight token tracker with Arc-owning RAII reservations.
//!
//! See `docs/superpowers/specs/2026-05-02-b5b-failure-mode-hardening-design.md`
//! Component 2 for design rationale.

#![allow(missing_docs)]
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::auth::TenantScope;
use crate::error::Error;

/// Snapshot of in-flight token usage for a single tenant.
///
/// `in_flight` is the total number of tokens currently reserved across all active
/// requests for the tenant. `high_water` is the all-time peak, useful for
/// capacity planning and alerting. Both values are updated atomically under the
/// `TenantTokenTracker`'s write lock.
#[derive(Debug, Default, Clone)]
pub struct TenantTokenState {
    pub in_flight: usize,
    pub high_water: usize,
}

/// Registry that enforces a per-tenant in-flight token cap across concurrent agent runs.
///
/// Each call to `reserve` atomically increments the tenant's `in_flight` counter
/// and returns a `TokenReservation`; the counter is decremented automatically when
/// the reservation is dropped. If the requested tokens would exceed `per_tenant_cap`,
/// `reserve` returns `Error::TenantOverloaded` rather than blocking, enabling
/// load-shedding at the orchestration layer. Pass an `Arc<TenantTokenTracker>` to
/// `OrchestratorBuilder::tenant_tracker` so sub-agents also participate in the cap.
pub struct TenantTokenTracker {
    states: RwLock<HashMap<String, TenantTokenState>>,
    per_tenant_cap: usize,
}

/// RAII guard that holds a token reservation for a tenant.
///
/// Created by `TenantTokenTracker::reserve` and automatically releases the
/// reserved token count back to the tracker when dropped, whether the request
/// succeeds, fails, or is cancelled. Callers should hold the reservation for
/// the duration of the LLM call to ensure the in-flight counter stays accurate.
pub struct TokenReservation {
    tracker: Arc<TenantTokenTracker>,
    tenant_id: String,
    tokens: usize,
}

impl std::fmt::Debug for TokenReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenReservation")
            .field("tenant_id", &self.tenant_id)
            .field("tokens", &self.tokens)
            .finish()
    }
}

impl Drop for TokenReservation {
    fn drop(&mut self) {
        self.tracker.release(&self.tenant_id, self.tokens);
    }
}

impl TenantTokenTracker {
    pub fn new(per_tenant_cap: usize) -> Self {
        Self {
            states: RwLock::new(HashMap::new()),
            per_tenant_cap,
        }
    }

    pub fn reserve(
        self: &Arc<Self>,
        scope: &TenantScope,
        tokens: usize,
    ) -> Result<TokenReservation, Error> {
        let tenant = scope.tenant_id.clone();
        let mut guard = self
            .states
            .write()
            .map_err(|_| Error::Agent("token tracker poisoned".into()))?;
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

    /// Adjust the in-flight token count for `scope` by `delta` (signed).
    ///
    /// Used by `AgentRunner` after each LLM response to reconcile the
    /// initial estimate with actual usage. Best-effort:
    /// - Positive deltas are silently **clamped at `per_tenant_cap`** to
    ///   avoid the per-turn caller having to handle a "we already accepted
    ///   this work" error mid-task. The clamp is intentional accounting
    ///   drift: if a tenant overshoots its cap during execution, the
    ///   tracker reports `in_flight == cap` rather than the true usage.
    ///   Subsequent `reserve()` calls will see the clamped value and
    ///   gate accordingly.
    /// - Negative deltas saturate at 0.
    /// - No-op on poisoned lock (logged via `tracing::warn!`) or unknown tenant.
    pub fn adjust(&self, scope: &TenantScope, delta: i64) {
        let mut guard = match self.states.write() {
            Ok(g) => g,
            Err(_) => {
                tracing::warn!(
                    tenant_id = %scope.tenant_id,
                    "token tracker poisoned during adjust; skipping reconciliation"
                );
                return;
            }
        };
        let Some(state) = guard.get_mut(&scope.tenant_id) else {
            return;
        };
        if delta >= 0 {
            state.in_flight = state
                .in_flight
                .saturating_add(delta as usize)
                .min(self.per_tenant_cap);
        } else {
            // `i64::unsigned_abs()` returns `u64` and handles `i64::MIN`
            // correctly; `-i64::MIN` would otherwise overflow.
            state.in_flight = state
                .in_flight
                .saturating_sub(delta.unsigned_abs() as usize);
        }
        if state.in_flight > state.high_water {
            state.high_water = state.in_flight;
        }
    }

    fn release(&self, tenant_id: &str, tokens: usize) {
        if let Ok(mut guard) = self.states.write()
            && let Some(state) = guard.get_mut(tenant_id)
        {
            state.in_flight = state.in_flight.saturating_sub(tokens);
        }
    }

    pub fn snapshot(&self) -> Vec<(String, TenantTokenState)> {
        match self.states.read() {
            Ok(g) => g.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            Err(_) => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(t: &str) -> TenantScope {
        TenantScope::new(t)
    }

    #[test]
    fn reserve_within_cap_succeeds() {
        let t = Arc::new(TenantTokenTracker::new(1000));
        let r = t.reserve(&scope("a"), 500).unwrap();
        let snap = t.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].1.in_flight, 500);
        drop(r);
    }

    #[test]
    fn reserve_exceeding_cap_returns_tenant_overloaded() {
        let t = Arc::new(TenantTokenTracker::new(100));
        let _r = t.reserve(&scope("a"), 80).unwrap();
        let err = t.reserve(&scope("a"), 50).unwrap_err();
        match err {
            Error::TenantOverloaded {
                tenant_id,
                in_flight,
                cap,
            } => {
                assert_eq!(tenant_id, "a");
                assert_eq!(in_flight, 80);
                assert_eq!(cap, 100);
            }
            other => panic!("expected TenantOverloaded, got {other:?}"),
        }
    }

    #[test]
    fn drop_releases_reservation() {
        let t = Arc::new(TenantTokenTracker::new(1000));
        {
            let _r = t.reserve(&scope("a"), 700).unwrap();
            assert_eq!(t.snapshot()[0].1.in_flight, 700);
        }
        assert_eq!(t.snapshot()[0].1.in_flight, 0);
    }

    #[test]
    fn tenants_are_isolated() {
        let t = Arc::new(TenantTokenTracker::new(100));
        let _ra = t.reserve(&scope("a"), 90).unwrap();
        let _rb = t.reserve(&scope("b"), 90).unwrap();
        let snap: HashMap<_, _> = t.snapshot().into_iter().collect();
        assert_eq!(snap["a"].in_flight, 90);
        assert_eq!(snap["b"].in_flight, 90);
    }

    #[test]
    fn high_water_tracks_peak() {
        let t = Arc::new(TenantTokenTracker::new(1000));
        let r1 = t.reserve(&scope("a"), 400).unwrap();
        let r2 = t.reserve(&scope("a"), 300).unwrap();
        drop(r1);
        let snap = t.snapshot();
        assert_eq!(snap[0].1.in_flight, 300);
        assert_eq!(snap[0].1.high_water, 700);
        drop(r2);
    }

    #[test]
    fn adjust_positive_delta_clamps_at_cap() {
        let t = Arc::new(TenantTokenTracker::new(1000));
        let _r = t.reserve(&scope("a"), 500).unwrap();
        t.adjust(&scope("a"), 800);
        let snap = t.snapshot();
        assert_eq!(snap[0].1.in_flight, 1000); // clamped
        assert_eq!(snap[0].1.high_water, 1000);
    }

    #[test]
    fn adjust_negative_delta_decrements() {
        let t = Arc::new(TenantTokenTracker::new(1000));
        let _r = t.reserve(&scope("a"), 500).unwrap();
        t.adjust(&scope("a"), -200);
        assert_eq!(t.snapshot()[0].1.in_flight, 300);
    }

    #[test]
    fn adjust_negative_i64_min_does_not_panic() {
        let t = Arc::new(TenantTokenTracker::new(1000));
        let _r = t.reserve(&scope("a"), 500).unwrap();
        t.adjust(&scope("a"), i64::MIN); // must not panic in debug
        assert_eq!(t.snapshot()[0].1.in_flight, 0);
    }

    #[tokio::test]
    async fn reservation_owns_arc_and_outlives_borrow() {
        // Compile-time check: TokenReservation can be moved into a future.
        let t = Arc::new(TenantTokenTracker::new(1000));
        let r = t.reserve(&scope("a"), 500).unwrap();
        let handle: tokio::task::JoinHandle<()> = tokio::task::spawn_blocking(move || {
            drop(r);
        });
        handle.await.unwrap();
    }

    #[test]
    fn default_scope_uses_empty_string_bucket() {
        let t = Arc::new(TenantTokenTracker::new(1000));
        let _r = t.reserve(&TenantScope::default(), 500).unwrap();
        let snap = t.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].0, ""); // empty-string sentinel
    }

    #[test]
    fn adjust_on_unknown_tenant_is_noop() {
        let t = Arc::new(TenantTokenTracker::new(1000));
        t.adjust(&scope("unknown"), -100);
        assert!(t.snapshot().is_empty());
    }
}
