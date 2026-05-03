//! Unix signal handling utilities for graceful agent shutdown.

use tokio_util::sync::CancellationToken;

/// Spawn a task that cancels `token` on SIGINT or SIGTERM.
pub fn spawn_shutdown_handler(token: CancellationToken) {
    tokio::spawn(async move {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigint = signal(SignalKind::interrupt()).expect("SIGINT handler");
        let mut sigterm = signal(SignalKind::terminate()).expect("SIGTERM handler");
        tokio::select! {
            _ = sigint.recv() => tracing::info!("SIGINT received"),
            _ = sigterm.recv() => tracing::info!("SIGTERM received"),
        }
        token.cancel();
    });
}
