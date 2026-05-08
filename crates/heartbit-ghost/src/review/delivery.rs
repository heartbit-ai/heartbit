//! Trait + types abstracting the user-interaction layer for review-mode
//! pipelines. `heartbit-ghost` doesn't know or care about Telegram —
//! production wires `TelegramReviewDelivery` from `heartbit-cli`; tests
//! wire `MockReviewDelivery`.

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

/// Input to a review delivery — the rendered message + correlation id.
#[derive(Debug, Clone)]
pub struct ReviewMessage {
    /// Persona instance name (rendered in the header).
    pub persona_name: String,
    /// Topic / shortname (rendered in the header).
    pub topic: String,
    /// Pre-rendered candidate drafts (one entry per surviving candidate).
    /// The delivery layer is responsible for laying these out (e.g.,
    /// numbered list with emoji indicators).
    pub candidates: Vec<String>,
    /// UUID for keyboard callback correlation. Used by the delivery
    /// implementation to thread callbacks back to the right pending review.
    pub interaction_id: Uuid,
}

/// Result of `ReviewDelivery::deliver_and_await`.
#[derive(Debug)]
pub struct DeliveredReview {
    /// What the user did (or didn't).
    pub outcome: DeliveryOutcome,
    /// Opaque ticket the impl uses to correlate `report()` back to this
    /// delivery. Telegram impl puts `{"chat_id": <i64>, "message_id": <i32>}`;
    /// mock can put `null`.
    pub receipt: DeliveryReceipt,
}

/// Opaque payload returned by `deliver_and_await`, threaded back to `report`.
#[derive(Debug, Clone)]
pub struct DeliveryReceipt {
    /// Implementation-defined data. Format is a contract between the
    /// concrete `ReviewDelivery` impl and itself; `run_review_pipeline`
    /// treats it as opaque.
    pub data: Value,
}

/// What the user did.
#[derive(Debug, Clone, PartialEq)]
pub enum DeliveryOutcome {
    /// User picked a specific candidate by 0-based index.
    Pick(usize),
    /// User pressed Skip.
    Skip,
    /// Timeout reached without a response.
    TimedOut,
}

/// What the orchestrator wants to report back to the user (via the
/// delivery layer's `report()` method, which typically edits the
/// original message in place).
#[derive(Debug, Clone)]
pub enum ReportableOutcome {
    /// Pick succeeded and tweet was posted.
    Posted {
        /// 0-based index into the original candidates list.
        chosen_index: usize,
        /// First-tweet URL.
        tweet_url: String,
    },
    /// User pressed Skip.
    Skipped,
    /// Timeout elapsed.
    TimedOut,
    /// Pick succeeded but `publish_gate` rejected the chosen draft.
    GateRejected {
        /// 0-based index of the rejected draft.
        chosen_index: usize,
        /// Reason from `PublishGateError`'s display.
        reason: String,
    },
    /// Pick succeeded, gate passed, but the X API call failed.
    PublishFailed {
        /// 0-based index of the draft that failed to post.
        chosen_index: usize,
        /// Failure reason (typically the X API error message).
        reason: String,
    },
}

/// Errors raised by the delivery layer.
#[derive(Debug, Error)]
pub enum ReviewDeliveryError {
    /// Bot connection / send / API failure.
    #[error("delivery transport: {0}")]
    Transport(String),
    /// Pick callback was received but couldn't be parsed.
    #[error("invalid callback: {0}")]
    InvalidCallback(String),
    /// Configuration failure (e.g., missing env vars).
    #[error("delivery config: {0}")]
    Config(String),
}

/// Object-safe async trait for delivering candidates to a user and
/// awaiting their pick.
///
/// Methods use the project's `Pin<Box<dyn Future>>` desugaring (matches
/// `heartbit_core::CredentialResolver`).
pub trait ReviewDelivery: Send + Sync {
    /// Send candidates to the user, wait for their pick (or timeout).
    /// Returns the outcome + opaque receipt for a later `report()`.
    fn deliver_and_await<'a>(
        &'a self,
        message: &'a ReviewMessage,
    ) -> Pin<Box<dyn Future<Output = Result<DeliveredReview, ReviewDeliveryError>> + Send + 'a>>;

    /// Update the prior delivery with the final result. Implementations
    /// may noop if the medium doesn't support editing. Failure is
    /// non-fatal at the caller (run_review_pipeline logs and continues).
    fn report<'a>(
        &'a self,
        receipt: DeliveryReceipt,
        outcome: ReportableOutcome,
    ) -> Pin<Box<dyn Future<Output = Result<(), ReviewDeliveryError>> + Send + 'a>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delivery_outcome_pick_displays_via_debug() {
        let o = DeliveryOutcome::Pick(2);
        assert_eq!(format!("{:?}", o), "Pick(2)");
    }

    #[test]
    fn reportable_outcome_posted_carries_url() {
        let o = ReportableOutcome::Posted {
            chosen_index: 1,
            tweet_url: "https://twitter.com/i/web/status/123".to_string(),
        };
        let s = format!("{:?}", o);
        assert!(s.contains("chosen_index: 1"), "got: {s}");
        assert!(s.contains("https://twitter.com"), "got: {s}");
    }

    #[test]
    fn review_delivery_error_transport_renders_inner_message() {
        let e = ReviewDeliveryError::Transport("connection refused".to_string());
        let s = format!("{e}");
        assert!(s.contains("connection refused"), "got: {s}");
        assert!(s.starts_with("delivery transport"), "got: {s}");
    }
}
