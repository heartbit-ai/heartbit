//! Wait-for-stability / network-idle settle (capability 6 of the browser-bot spec).
//!
//! Playwright's actionability contract is the reference: never act on a
//! half-rendered page. `chrome-devtools-mcp`'s `wait_for` only resolves on a
//! *specific expected string*, so it cannot express "wait until the page stops
//! changing". Acting on an unsettled SPA poisons the whole observe→act→verify
//! loop (a `uid` grounded on a mid-render snapshot is stale before the click).
//!
//! This module synthesizes a settle primitive from a cheap, repeatable probe:
//! `document.readyState === "complete"` plus a DOM-content signature that must
//! stop changing for a configurable idle window. The decision logic is a **pure
//! state machine** ([`step`]) folding one [`Probe`] at a time with a
//! caller-supplied `dt` — no wall-clock — so every transition is exhaustively
//! unit-testable. The async [`settle`] driver is thin glue over it: call probe,
//! measure elapsed, fold, sleep, repeat.
//!
//! Network-idle (counting in-flight requests via `list_network_requests`) is a
//! planned extension; its live output shape is not yet captured, and guessing it
//! would be the exact fabrication this project guards against. v1 settles on
//! DOM-ready + DOM-stability, which already covers the dominant SPA-render case.

use std::future::Future;
use std::time::{Duration, Instant};

use crate::error::Error;

/// Tuning for [`settle`].
#[derive(Debug, Clone)]
pub struct SettleConfig {
    /// Hard upper bound on the total wait. On reaching it, [`settle`] returns
    /// [`SettleOutcome::TimedOut`] regardless of quiescence.
    pub timeout: Duration,
    /// How long the page must remain continuously quiescent (DOM-ready and its
    /// content signature unchanged) before it is considered settled.
    pub idle_window: Duration,
    /// Delay between successive probes.
    pub poll_interval: Duration,
}

impl Default for SettleConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(10),
            idle_window: Duration::from_millis(500),
            poll_interval: Duration::from_millis(100),
        }
    }
}

/// One probe reading of page quiescence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Probe {
    /// `document.readyState === "complete"`.
    pub dom_ready: bool,
    /// A cheap signature of DOM content (e.g. serialized-length or node count).
    /// Compared only for *equality* across consecutive probes to detect ongoing
    /// mutation; its absolute value is meaningless.
    pub dom_signature: u64,
}

/// Terminal result of [`settle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettleOutcome {
    /// The page was DOM-ready and unchanged for the full idle window.
    Settled,
    /// The timeout elapsed before the page settled.
    TimedOut,
}

/// Running accumulator threaded across probes (pure; carries no clock).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SettleState {
    total: Duration,
    quiet: Duration,
    last_sig: Option<u64>,
}

impl SettleState {
    /// The initial state before any probe.
    pub(crate) fn start() -> Self {
        Self {
            total: Duration::ZERO,
            quiet: Duration::ZERO,
            last_sig: None,
        }
    }
}

/// The result of folding one probe into the running state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SettleStep {
    /// Not settled yet; carry this state into the next probe.
    KeepWaiting(SettleState),
    /// Reached a terminal outcome.
    Done(SettleOutcome),
}

/// Pure transition: fold one [`Probe`] — taken `dt` after the previous one —
/// into `st`, yielding the next state or a terminal [`SettleOutcome`].
///
/// A probe counts as *quiet* only when the page is DOM-ready **and** its content
/// signature is unchanged from the previous probe. The very first probe is never
/// quiet (no prior signature to compare), so settling always requires at least
/// two consecutive matching readings. Settling is checked before the timeout, so
/// a page that goes quiet on the same probe that crosses the deadline still
/// settles.
pub(crate) fn step(cfg: &SettleConfig, mut st: SettleState, dt: Duration, p: Probe) -> SettleStep {
    st.total = st.total.saturating_add(dt);

    let unchanged = st.last_sig == Some(p.dom_signature);
    let quiet_now = p.dom_ready && unchanged;
    if quiet_now {
        st.quiet = st.quiet.saturating_add(dt);
    } else {
        st.quiet = Duration::ZERO;
    }
    st.last_sig = Some(p.dom_signature);

    // Settling is checked BEFORE the timeout so that a probe which both completes
    // the idle window and crosses the deadline reports Settled, not TimedOut.
    if st.quiet >= cfg.idle_window {
        SettleStep::Done(SettleOutcome::Settled)
    } else if st.total >= cfg.timeout {
        SettleStep::Done(SettleOutcome::TimedOut)
    } else {
        SettleStep::KeepWaiting(st)
    }
}

/// Drive a settle loop: repeatedly call `probe`, folding each reading via
/// [`step`] and sleeping `poll_interval` between calls, until the page settles
/// or the timeout elapses. `dt` for each fold is the real elapsed time since the
/// previous probe, so the loop is self-pacing.
pub async fn settle<F, Fut>(cfg: &SettleConfig, mut probe: F) -> Result<SettleOutcome, Error>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<Probe, Error>>,
{
    let mut st = SettleState::start();
    let mut last = Instant::now();
    loop {
        let p = probe().await?;
        let now = Instant::now();
        let dt = now.saturating_duration_since(last);
        last = now;
        match step(cfg, st, dt, p) {
            SettleStep::Done(outcome) => return Ok(outcome),
            SettleStep::KeepWaiting(next) => st = next,
        }
        tokio::time::sleep(cfg.poll_interval).await;
    }
}

/// Interpret the output of `evaluate_script(() => document.readyState)` as
/// DOM-readiness. chrome-devtools-mcp returns the JSON result wrapped in text
/// (verified live: a fenced ```json block containing `"complete"`); only the
/// `"complete"` readyState is treated as ready (`"loading"`/`"interactive"` are
/// not). Matching the quoted token avoids false positives from surrounding prose.
pub fn parse_dom_ready(eval_output: &str) -> bool {
    eval_output.contains("\"complete\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(timeout_ms: u64, idle_ms: u64) -> SettleConfig {
        SettleConfig {
            timeout: Duration::from_millis(timeout_ms),
            idle_window: Duration::from_millis(idle_ms),
            poll_interval: Duration::ZERO,
        }
    }

    fn ready(sig: u64) -> Probe {
        Probe {
            dom_ready: true,
            dom_signature: sig,
        }
    }
    fn loading(sig: u64) -> Probe {
        Probe {
            dom_ready: false,
            dom_signature: sig,
        }
    }

    // --- pure step() state machine (deterministic, no clock) ---

    #[test]
    fn first_probe_is_never_quiet() {
        // No prior signature to compare → cannot be "unchanged" → quiet stays 0.
        let c = cfg(1000, 500);
        match step(&c, SettleState::start(), Duration::ZERO, ready(5)) {
            SettleStep::KeepWaiting(st) => assert_eq!(st.quiet, Duration::ZERO),
            other => panic!("first probe must keep waiting, got {other:?}"),
        }
    }

    #[test]
    fn stable_ready_accumulates_quiet_then_settles() {
        let c = cfg(10_000, 500);
        // 1st probe establishes the signature (quiet=0).
        let st = match step(
            &c,
            SettleState::start(),
            Duration::from_millis(100),
            ready(5),
        ) {
            SettleStep::KeepWaiting(st) => st,
            other => panic!("expected KeepWaiting, got {other:?}"),
        };
        // 2nd probe: unchanged + ready → quiet accrues 300ms (< 500 window).
        let st = match step(&c, st, Duration::from_millis(300), ready(5)) {
            SettleStep::KeepWaiting(st) => st,
            other => panic!("expected KeepWaiting at 300ms quiet, got {other:?}"),
        };
        // 3rd probe: another 300ms quiet → 600ms ≥ 500ms window → Settled.
        assert_eq!(
            step(&c, st, Duration::from_millis(300), ready(5)),
            SettleStep::Done(SettleOutcome::Settled)
        );
    }

    #[test]
    fn dom_mutation_resets_quiet() {
        let c = cfg(10_000, 500);
        let st = match step(
            &c,
            SettleState::start(),
            Duration::from_millis(100),
            ready(5),
        ) {
            SettleStep::KeepWaiting(st) => st,
            o => panic!("{o:?}"),
        };
        let st = match step(&c, st, Duration::from_millis(400), ready(5)) {
            SettleStep::KeepWaiting(st) => st,
            o => panic!("{o:?}"),
        };
        // Signature changes (DOM still mutating) → quiet resets to 0.
        match step(&c, st, Duration::from_millis(100), ready(9)) {
            SettleStep::KeepWaiting(st) => assert_eq!(st.quiet, Duration::ZERO),
            o => panic!("mutation must reset quiet, got {o:?}"),
        }
    }

    #[test]
    fn not_ready_resets_quiet() {
        let c = cfg(10_000, 500);
        let st = match step(
            &c,
            SettleState::start(),
            Duration::from_millis(100),
            ready(5),
        ) {
            SettleStep::KeepWaiting(st) => st,
            o => panic!("{o:?}"),
        };
        let st = match step(&c, st, Duration::from_millis(400), ready(5)) {
            SettleStep::KeepWaiting(st) => st,
            o => panic!("{o:?}"),
        };
        // readyState regresses to loading (same sig) → not quiet → reset.
        match step(&c, st, Duration::from_millis(100), loading(5)) {
            SettleStep::KeepWaiting(st) => assert_eq!(st.quiet, Duration::ZERO),
            o => panic!("not-ready must reset quiet, got {o:?}"),
        }
    }

    #[test]
    fn times_out_when_never_quiet() {
        let c = cfg(1000, 500);
        // A page that keeps mutating: each probe has a new signature → never quiet.
        let mut st = SettleState::start();
        let mut sig = 0;
        let mut outcome = None;
        for _ in 0..20 {
            sig += 1;
            match step(&c, st, Duration::from_millis(200), ready(sig)) {
                SettleStep::Done(o) => {
                    outcome = Some(o);
                    break;
                }
                SettleStep::KeepWaiting(next) => st = next,
            }
        }
        assert_eq!(outcome, Some(SettleOutcome::TimedOut));
    }

    #[test]
    fn settle_takes_priority_over_timeout_on_same_probe() {
        // total will cross the 1000ms timeout on the SAME probe that completes the
        // idle window; settling must win.
        let c = cfg(1000, 500);
        let st = SettleState {
            total: Duration::from_millis(900),
            quiet: Duration::from_millis(400),
            last_sig: Some(5),
        };
        // dt=200ms → total=1100 (≥timeout) AND quiet=600 (≥window). Settled wins.
        assert_eq!(
            step(&c, st, Duration::from_millis(200), ready(5)),
            SettleStep::Done(SettleOutcome::Settled)
        );
    }

    // --- async settle() driver (deterministic via zero-config edges) ---

    #[tokio::test]
    async fn settle_times_out_immediately_with_zero_timeout() {
        // timeout=0, idle_window>0, a busy page → first fold yields TimedOut,
        // no sleeps, no real-time dependence.
        let c = cfg(0, 1000);
        let out = settle(&c, || async { Ok(loading(1)) }).await.expect("ok");
        assert_eq!(out, SettleOutcome::TimedOut);
    }

    #[tokio::test]
    async fn settle_returns_settled_for_a_quiet_page() {
        // A page that is immediately ready and stable settles within the window.
        // idle_window small; the driver folds real (tiny) dt values until quiet
        // accrues. poll_interval=0 keeps it fast; correctness of timing is proven
        // by the pure step() tests — here we only assert the driver reaches it.
        let c = SettleConfig {
            timeout: Duration::from_secs(5),
            idle_window: Duration::from_nanos(1),
            poll_interval: Duration::ZERO,
        };
        let out = settle(&c, || async { Ok(ready(7)) }).await.expect("ok");
        assert_eq!(out, SettleOutcome::Settled);
    }

    #[tokio::test]
    async fn settle_propagates_probe_error() {
        let c = SettleConfig::default();
        let r = settle(&c, || async { Err(Error::Agent("probe boom".into())) }).await;
        assert!(r.is_err(), "a probe error must propagate, not be swallowed");
    }

    // --- parse_dom_ready against the real captured output format ---

    #[test]
    fn parse_dom_ready_matches_only_complete() {
        // The exact wrapper chrome-devtools-mcp emits (captured live this session).
        let complete = "Script ran on page and returned:\n```json\n\"complete\"\n```";
        assert!(parse_dom_ready(complete));
        assert!(!parse_dom_ready(
            "Script ran on page and returned:\n```json\n\"loading\"\n```"
        ));
        assert!(!parse_dom_ready(
            "Script ran on page and returned:\n```json\n\"interactive\"\n```"
        ));
    }
}
