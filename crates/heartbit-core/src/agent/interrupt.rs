//! A re-armable, per-turn interrupt handle for an [`AgentRunner`](super::AgentRunner).
//!
//! A TUI (or any caller) can ask the agent to abandon the **current turn** —
//! aborting an in-flight LLM generation — and return to awaiting the next input,
//! WITHOUT tearing down the session (conversation history is preserved). Because
//! a [`CancellationToken`] is one-shot, this handle swaps in a fresh token after
//! each interrupt so the next turn starts armed again.
//!
//! Semantics (interactive path): on interrupt the runner abandons whatever the
//! turn is doing and ends it cleanly, then waits for the next message via
//! `on_input`. An interrupt during an LLM generation aborts the stream; an
//! interrupt during a tool batch abandons it — synthesizing a result for every
//! in-flight `tool_use` (so no call is left unanswered) and killing any
//! subprocess via `kill_on_drop` — then the next LLM-call race ends the turn.

use std::sync::{Arc, Mutex};

use tokio_util::sync::CancellationToken;

/// A cloneable handle to interrupt the in-flight turn of an agent run.
#[derive(Clone)]
pub struct InterruptHandle {
    inner: Arc<Mutex<CancellationToken>>,
}

impl Default for InterruptHandle {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(CancellationToken::new())),
        }
    }
}

impl InterruptHandle {
    /// Create a fresh, un-triggered handle.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request that the agent abandon its current turn. Idempotent until the
    /// runner [`rearm`](Self::rearm)s for the next turn.
    pub fn interrupt(&self) {
        self.inner.lock().expect("interrupt lock poisoned").cancel();
    }

    /// Whether the current turn's token has been triggered.
    pub fn is_interrupted(&self) -> bool {
        self.inner
            .lock()
            .expect("interrupt lock poisoned")
            .is_cancelled()
    }

    /// A clone of the current turn's token, to race an LLM call against.
    pub(crate) fn token(&self) -> CancellationToken {
        self.inner.lock().expect("interrupt lock poisoned").clone()
    }

    /// Swap in a fresh token, arming the handle for the next turn. Called by the
    /// runner right after it has handled an interrupt.
    pub(crate) fn rearm(&self) {
        *self.inner.lock().expect("interrupt lock poisoned") = CancellationToken::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_un_triggered() {
        assert!(!InterruptHandle::new().is_interrupted());
    }

    // Regression test for the live mid-tool-interrupt topology, minimized: a
    // `select!` racing the per-turn token against a real SUBPROCESS, spawned on a
    // JoinSet, on a SEPARATE thread with its OWN current_thread runtime (the
    // hardest flavor — one scheduler thread), cancelled CROSS-THREAD, with a
    // blocking sync-approval gate before the race. Proves the cancel arm wins
    // (returns "cancelled") — if it ever returns "completed", cross-runtime
    // interrupt over a subprocess is broken. Pairs with the end-to-end
    // `runner::tests::interrupt_during_tool_batch_abandons_it_and_ends_turn`.
    #[tokio::test(flavor = "multi_thread")]
    async fn cross_runtime_interrupt_aborts_subprocess_tool_race() {
        use std::time::Duration;
        // Force the MAIN (this) runtime to register tokio::process / SIGCHLD first,
        // exactly like a process that already has a primary runtime (the TUI).
        let _ = tokio::process::Command::new("true").status().await;

        let handle = InterruptHandle::new();
        let h2 = handle.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let (appr_tx, appr_rx) = std::sync::mpsc::channel::<()>();
        // Agent on its OWN current_thread runtime, on a separate thread (like the TUI).
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let outcome = rt.block_on(async move {
                appr_rx.recv().unwrap(); // mimic the synchronous on_approval block
                let token = h2.token();
                tokio::select! {
                    biased;
                    _ = token.cancelled() => "cancelled",
                    _ = async {
                        let mut js = tokio::task::JoinSet::new();
                        js.spawn(async {
                            // Mirror BashTool EXACTLY: piped stdout/stderr +
                            // wait_with_output wrapped in tokio::time::timeout.
                            let child = tokio::process::Command::new("sh")
                                .arg("-c").arg("sleep 5")
                                .stdin(std::process::Stdio::null())
                                .stdout(std::process::Stdio::piped())
                                .stderr(std::process::Stdio::piped())
                                .spawn();
                            if let Ok(child) = child {
                                let _ = tokio::time::timeout(
                                    Duration::from_secs(30),
                                    child.wait_with_output(),
                                )
                                .await;
                            }
                        });
                        js.join_next().await;
                    } => "completed",
                }
            });
            let _ = tx.send(outcome);
        });
        // Drive timing + the interrupt from a TASK on the MAIN runtime (like the UI).
        tokio::time::sleep(Duration::from_millis(150)).await;
        appr_tx.send(()).unwrap();
        tokio::time::sleep(Duration::from_millis(250)).await;
        handle.interrupt();
        let outcome = tokio::task::spawn_blocking(move || rx.recv_timeout(Duration::from_secs(8)))
            .await
            .unwrap()
            .expect("thread finished");
        assert_eq!(
            outcome, "cancelled",
            "cross-thread interrupt must abort the subprocess race"
        );
    }

    #[test]
    fn interrupt_triggers_then_rearm_clears() {
        let h = InterruptHandle::new();
        let before = h.token();
        h.interrupt();
        assert!(h.is_interrupted());
        assert!(
            before.is_cancelled(),
            "the live token reflects the interrupt"
        );
        h.rearm();
        assert!(!h.is_interrupted(), "rearm arms a fresh token");
        // The previously-handed-out token stays cancelled (it's the old turn's).
        assert!(before.is_cancelled());
    }

    #[test]
    fn clones_share_state() {
        let h = InterruptHandle::new();
        let clone = h.clone();
        clone.interrupt();
        assert!(h.is_interrupted(), "a clone interrupts the same run");
    }
}
