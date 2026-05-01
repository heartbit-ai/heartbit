use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use uuid::Uuid;

use crate::agent::OnInput;
use crate::agent::events::{AgentEvent, OnEvent};
use crate::error::Error;
use crate::llm::{ApprovalDecision, OnApproval, OnText};
use crate::tool::builtins::{OnQuestion, QuestionRequest, QuestionResponse};

/// Messages sent from bridge to the WS handler for forwarding to the client.
#[derive(Debug, Clone)]
pub enum OutboundMessage {
    TextDelta {
        session_id: Uuid,
        text: String,
    },
    AgentEvent {
        session_id: Uuid,
        event: AgentEvent,
    },
    InputNeeded {
        session_id: Uuid,
        interaction_id: Uuid,
    },
    ApprovalNeeded {
        session_id: Uuid,
        interaction_id: Uuid,
        tool_calls: serde_json::Value,
    },
    QuestionNeeded {
        session_id: Uuid,
        interaction_id: Uuid,
        request: QuestionRequest,
    },
    /// Agent run completed successfully.
    ChatFinal {
        session_id: Uuid,
        result: String,
    },
    /// Agent run failed with an error.
    ChatError {
        session_id: Uuid,
        error: String,
    },
    /// Pre-built WS frame (e.g., method responses) to send as-is.
    RawFrame(crate::channel::types::WsFrame),
}

/// Sender half for pending interactions.
enum PendingSender {
    Input(tokio::sync::oneshot::Sender<Option<String>>),
    Approval(std::sync::mpsc::Sender<ApprovalDecision>),
    Question(tokio::sync::oneshot::Sender<Result<QuestionResponse, Error>>),
}

/// Grace period after timeout before cleaning up a pending interaction entry.
/// During this window, a late resolve from the client succeeds silently
/// (the oneshot receiver is already dropped, but the resolve call returns Ok).
const GRACE_PERIOD: Duration = Duration::from_secs(15);

/// The interaction bridge translating heartbit callbacks to WS frames.
///
/// Maintains a map of pending interactions and an outbound channel for
/// pushing events to the WebSocket handler.
pub struct InteractionBridge {
    pending: RwLock<HashMap<Uuid, PendingSender>>,
    outbound: tokio::sync::mpsc::Sender<OutboundMessage>,
    timeout: Duration,
}

impl InteractionBridge {
    /// Create a new bridge with the given outbound sender and interaction timeout.
    pub fn new(outbound: tokio::sync::mpsc::Sender<OutboundMessage>, timeout: Duration) -> Self {
        Self {
            pending: RwLock::new(HashMap::new()),
            outbound,
            timeout,
        }
    }

    /// Create an `OnText` callback that forwards text deltas to the outbound channel.
    pub fn make_on_text(self: &Arc<Self>, session_id: Uuid) -> Arc<OnText> {
        let outbound = self.outbound.clone();
        Arc::new(move |text: &str| {
            let _ = outbound.try_send(OutboundMessage::TextDelta {
                session_id,
                text: text.to_string(),
            });
        })
    }

    /// Create an `OnEvent` callback that forwards agent events to the outbound channel.
    pub fn make_on_event(self: &Arc<Self>, session_id: Uuid) -> Arc<OnEvent> {
        let outbound = self.outbound.clone();
        Arc::new(move |event: AgentEvent| {
            let _ = outbound.try_send(OutboundMessage::AgentEvent { session_id, event });
        })
    }

    /// Create an `OnInput` callback using oneshot rendezvous.
    ///
    /// When called, sends `InputNeeded` on the outbound channel and waits
    /// for the client to resolve. Returns `None` on timeout (SOLICITATION semantic).
    pub fn make_on_input(self: &Arc<Self>, session_id: Uuid) -> Arc<OnInput> {
        let bridge = Arc::clone(self);
        Arc::new(move || {
            let bridge = Arc::clone(&bridge);
            Box::pin(async move {
                let interaction_id = Uuid::new_v4();
                let (tx, rx) = tokio::sync::oneshot::channel();

                // Store the sender
                {
                    let mut pending = bridge.pending.write().expect("pending lock not poisoned");
                    pending.insert(interaction_id, PendingSender::Input(tx));
                }

                // Send outbound event
                let _ = bridge.outbound.try_send(OutboundMessage::InputNeeded {
                    session_id,
                    interaction_id,
                });

                // Await with timeout
                match tokio::time::timeout(bridge.timeout, rx).await {
                    Ok(Ok(msg)) => msg,
                    _ => {
                        // Timeout or channel closed -> deferred cleanup (grace period
                        // allows late resolves from the client to succeed silently).
                        let cleanup_bridge = Arc::clone(&bridge);
                        tokio::spawn(async move {
                            tokio::time::sleep(GRACE_PERIOD).await;
                            cleanup_bridge.cleanup_pending(interaction_id);
                        });
                        None
                    }
                }
            })
        })
    }

    /// Create an `OnApproval` callback using `std::sync::mpsc` rendezvous.
    ///
    /// `OnApproval` is synchronous, so we use `std::sync::mpsc::recv_timeout()`
    /// instead of async oneshot. Returns `Deny` on timeout (PERMISSION semantic).
    pub fn make_on_approval(self: &Arc<Self>, session_id: Uuid) -> Arc<OnApproval> {
        let bridge = Arc::clone(self);
        Arc::new(move |tool_calls: &[crate::llm::types::ToolCall]| {
            let interaction_id = Uuid::new_v4();
            let (tx, rx) = std::sync::mpsc::channel();

            // Serialize tool calls for the outbound message
            let tool_calls_json = serde_json::to_value(tool_calls).unwrap_or_default();

            // Store the sender
            {
                let mut pending = bridge.pending.write().expect("pending lock not poisoned");
                pending.insert(interaction_id, PendingSender::Approval(tx));
            }

            // Send outbound event (non-blocking try_send)
            let _ = bridge.outbound.try_send(OutboundMessage::ApprovalNeeded {
                session_id,
                interaction_id,
                tool_calls: tool_calls_json,
            });

            // Block on recv_timeout
            match rx.recv_timeout(bridge.timeout) {
                Ok(decision) => decision,
                Err(_) => {
                    // Timeout or disconnected -> deny (safe default).
                    // Deferred cleanup: grace period allows late client resolves.
                    // Note: we're in a sync context so we spawn via Handle.
                    let cleanup_bridge = Arc::clone(&bridge);
                    if let Ok(handle) = tokio::runtime::Handle::try_current() {
                        handle.spawn(async move {
                            tokio::time::sleep(GRACE_PERIOD).await;
                            cleanup_bridge.cleanup_pending(interaction_id);
                        });
                    } else {
                        bridge.cleanup_pending(interaction_id);
                    }
                    ApprovalDecision::Deny
                }
            }
        })
    }

    /// Create an `OnQuestion` callback using oneshot rendezvous.
    ///
    /// Returns `Error::Channel("timeout")` on timeout (CLARIFICATION semantic).
    pub fn make_on_question(self: &Arc<Self>, session_id: Uuid) -> Arc<OnQuestion> {
        let bridge = Arc::clone(self);
        Arc::new(move |request: QuestionRequest| {
            let bridge = Arc::clone(&bridge);
            let request_clone = request.clone();
            Box::pin(async move {
                let interaction_id = Uuid::new_v4();
                let (tx, rx) = tokio::sync::oneshot::channel();

                // Store the sender
                {
                    let mut pending = bridge.pending.write().expect("pending lock not poisoned");
                    pending.insert(interaction_id, PendingSender::Question(tx));
                }

                // Send outbound event
                let _ = bridge.outbound.try_send(OutboundMessage::QuestionNeeded {
                    session_id,
                    interaction_id,
                    request: request_clone,
                });

                // Await with timeout
                match tokio::time::timeout(bridge.timeout, rx).await {
                    Ok(Ok(result)) => result,
                    Ok(Err(_)) => {
                        // Channel closed — deferred cleanup with grace period.
                        let cleanup_bridge = Arc::clone(&bridge);
                        tokio::spawn(async move {
                            tokio::time::sleep(GRACE_PERIOD).await;
                            cleanup_bridge.cleanup_pending(interaction_id);
                        });
                        Err(Error::Channel("interaction channel closed".into()))
                    }
                    Err(_) => {
                        // Timeout — deferred cleanup with grace period.
                        let cleanup_bridge = Arc::clone(&bridge);
                        tokio::spawn(async move {
                            tokio::time::sleep(GRACE_PERIOD).await;
                            cleanup_bridge.cleanup_pending(interaction_id);
                        });
                        Err(Error::Channel("interaction timed out".into()))
                    }
                }
            })
        })
    }

    // --- Resolve methods ---

    /// Resolve a pending input interaction.
    pub fn resolve_input(&self, id: Uuid, message: Option<String>) -> Result<(), Error> {
        let sender = self.take_pending(id)?;
        match sender {
            PendingSender::Input(tx) => {
                let _ = tx.send(message);
                Ok(())
            }
            other => {
                drop(other);
                Err(Error::Channel(format!(
                    "interaction {id} is not an input interaction"
                )))
            }
        }
    }

    /// Resolve a pending approval interaction.
    pub fn resolve_approval(&self, id: Uuid, decision: ApprovalDecision) -> Result<(), Error> {
        let sender = self.take_pending(id)?;
        match sender {
            PendingSender::Approval(tx) => {
                let _ = tx.send(decision);
                Ok(())
            }
            other => {
                drop(other);
                Err(Error::Channel(format!(
                    "interaction {id} is not an approval interaction"
                )))
            }
        }
    }

    /// Resolve a pending question interaction.
    pub fn resolve_question(&self, id: Uuid, response: QuestionResponse) -> Result<(), Error> {
        let sender = self.take_pending(id)?;
        match sender {
            PendingSender::Question(tx) => {
                let _ = tx.send(Ok(response));
                Ok(())
            }
            other => {
                drop(other);
                Err(Error::Channel(format!(
                    "interaction {id} is not a question interaction"
                )))
            }
        }
    }

    /// Remove and return a pending interaction, or error if not found.
    fn take_pending(&self, id: Uuid) -> Result<PendingSender, Error> {
        let mut pending = self
            .pending
            .write()
            .map_err(|e| Error::Channel(format!("lock poisoned: {e}")))?;
        pending
            .remove(&id)
            .ok_or_else(|| Error::Channel(format!("no pending interaction with id {id}")))
    }

    /// Remove a pending interaction without error (cleanup after timeout).
    fn cleanup_pending(&self, id: Uuid) {
        if let Ok(mut pending) = self.pending.write() {
            pending.remove(&id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tool::builtins::{Question, QuestionOption};

    fn make_bridge(
        timeout: Duration,
    ) -> (
        Arc<InteractionBridge>,
        tokio::sync::mpsc::Receiver<OutboundMessage>,
    ) {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let bridge = Arc::new(InteractionBridge::new(tx, timeout));
        (bridge, rx)
    }

    fn make_question_request() -> QuestionRequest {
        QuestionRequest {
            questions: vec![Question {
                question: "Pick a color".into(),
                header: "Color".into(),
                options: vec![
                    QuestionOption {
                        label: "Red".into(),
                        description: "A warm color".into(),
                    },
                    QuestionOption {
                        label: "Blue".into(),
                        description: "A cool color".into(),
                    },
                ],
                multiple: false,
            }],
        }
    }

    // --- 1. text_delta_forwarded ---

    #[tokio::test]
    async fn text_delta_forwarded() {
        let (bridge, mut rx) = make_bridge(Duration::from_secs(5));
        let session_id = Uuid::new_v4();
        let on_text = bridge.make_on_text(session_id);

        on_text("hello world");

        let msg = rx.recv().await.expect("should receive outbound message");
        match msg {
            OutboundMessage::TextDelta {
                session_id: sid,
                text,
            } => {
                assert_eq!(sid, session_id);
                assert_eq!(text, "hello world");
            }
            other => panic!("expected TextDelta, got: {other:?}"),
        }
    }

    // --- 2. agent_event_forwarded ---

    #[tokio::test]
    async fn agent_event_forwarded() {
        let (bridge, mut rx) = make_bridge(Duration::from_secs(5));
        let session_id = Uuid::new_v4();
        let on_event = bridge.make_on_event(session_id);

        let event = AgentEvent::RunStarted {
            agent: "test".into(),
            task: "do stuff".into(),
        };
        on_event(event.clone());

        let msg = rx.recv().await.expect("should receive outbound message");
        match msg {
            OutboundMessage::AgentEvent {
                session_id: sid,
                event: received,
            } => {
                assert_eq!(sid, session_id);
                // Verify the event roundtripped via serde
                let expected_json = serde_json::to_string(&AgentEvent::RunStarted {
                    agent: "test".into(),
                    task: "do stuff".into(),
                })
                .expect("serialize");
                let received_json = serde_json::to_string(&received).expect("serialize");
                assert_eq!(expected_json, received_json);
            }
            other => panic!("expected AgentEvent, got: {other:?}"),
        }
    }

    // --- 3. resolve_input_before_timeout ---

    #[tokio::test]
    async fn resolve_input_before_timeout() {
        let (bridge, mut rx) = make_bridge(Duration::from_secs(5));
        let session_id = Uuid::new_v4();
        let on_input = bridge.make_on_input(session_id);

        // Spawn the callback
        let handle = tokio::spawn(async move { on_input().await });

        // Wait for the outbound message
        let msg = rx.recv().await.expect("should receive InputNeeded");
        let interaction_id = match msg {
            OutboundMessage::InputNeeded { interaction_id, .. } => interaction_id,
            other => panic!("expected InputNeeded, got: {other:?}"),
        };

        // Resolve
        bridge
            .resolve_input(interaction_id, Some("hello".into()))
            .expect("resolve should succeed");

        // Verify
        let result = handle.await.expect("task should complete");
        assert_eq!(result, Some("hello".into()));
    }

    // --- 4. input_timeout_returns_none ---

    #[tokio::test]
    async fn input_timeout_returns_none() {
        let (bridge, mut rx) = make_bridge(Duration::from_millis(10));
        let session_id = Uuid::new_v4();
        let on_input = bridge.make_on_input(session_id);

        let handle = tokio::spawn(async move { on_input().await });

        // Consume the outbound message but don't resolve
        let _msg = rx.recv().await.expect("should receive InputNeeded");

        // Verify timeout returns None
        let result = handle.await.expect("task should complete");
        assert_eq!(result, None);
    }

    // --- 5. resolve_approval_before_timeout ---

    #[tokio::test]
    async fn resolve_approval_before_timeout() {
        let (bridge, mut rx) = make_bridge(Duration::from_secs(5));
        let session_id = Uuid::new_v4();
        let on_approval = bridge.make_on_approval(session_id);

        // Spawn on blocking thread (OnApproval is sync + blocks)
        let handle = tokio::task::spawn_blocking(move || on_approval(&[]));

        // Wait for outbound message
        let msg = rx.recv().await.expect("should receive ApprovalNeeded");
        let interaction_id = match msg {
            OutboundMessage::ApprovalNeeded { interaction_id, .. } => interaction_id,
            other => panic!("expected ApprovalNeeded, got: {other:?}"),
        };

        // Resolve
        bridge
            .resolve_approval(interaction_id, ApprovalDecision::Allow)
            .expect("resolve should succeed");

        // Verify
        let result = handle.await.expect("task should complete");
        assert_eq!(result, ApprovalDecision::Allow);
    }

    // --- 6. approval_timeout_returns_deny ---

    #[tokio::test]
    async fn approval_timeout_returns_deny() {
        let (bridge, mut rx) = make_bridge(Duration::from_millis(10));
        let session_id = Uuid::new_v4();
        let on_approval = bridge.make_on_approval(session_id);

        let handle = tokio::task::spawn_blocking(move || on_approval(&[]));

        // Consume outbound but don't resolve
        let _msg = rx.recv().await.expect("should receive ApprovalNeeded");

        // Verify timeout returns Deny
        let result = handle.await.expect("task should complete");
        assert_eq!(result, ApprovalDecision::Deny);
    }

    // --- 7. resolve_question_before_timeout ---

    #[tokio::test]
    async fn resolve_question_before_timeout() {
        let (bridge, mut rx) = make_bridge(Duration::from_secs(5));
        let session_id = Uuid::new_v4();
        let on_question = bridge.make_on_question(session_id);

        let request = make_question_request();
        let handle = tokio::spawn(async move { on_question(request).await });

        // Wait for outbound message
        let msg = rx.recv().await.expect("should receive QuestionNeeded");
        let interaction_id = match msg {
            OutboundMessage::QuestionNeeded { interaction_id, .. } => interaction_id,
            other => panic!("expected QuestionNeeded, got: {other:?}"),
        };

        // Resolve
        let response = QuestionResponse {
            answers: vec![vec!["Red".into()]],
        };
        bridge
            .resolve_question(interaction_id, response)
            .expect("resolve should succeed");

        // Verify
        let result = handle.await.expect("task should complete");
        let resp = result.expect("should be Ok");
        assert_eq!(resp.answers, vec![vec!["Red".to_string()]]);
    }

    // --- 8. question_timeout_returns_error ---

    #[tokio::test]
    async fn question_timeout_returns_error() {
        let (bridge, mut rx) = make_bridge(Duration::from_millis(10));
        let session_id = Uuid::new_v4();
        let on_question = bridge.make_on_question(session_id);

        let request = make_question_request();
        let handle = tokio::spawn(async move { on_question(request).await });

        // Consume outbound but don't resolve
        let _msg = rx.recv().await.expect("should receive QuestionNeeded");

        // Verify timeout returns Err(Channel)
        let result = handle.await.expect("task should complete");
        let err = result.expect_err("should be Err");
        assert!(
            err.to_string().contains("timed out"),
            "error should mention timeout, got: {err}"
        );
    }

    // --- 9. resolve_unknown_id_returns_error ---

    #[tokio::test]
    async fn resolve_unknown_id_returns_error() {
        let (bridge, _rx) = make_bridge(Duration::from_secs(5));
        let bogus_id = Uuid::new_v4();

        let err = bridge
            .resolve_input(bogus_id, Some("msg".into()))
            .expect_err("should error");
        assert!(
            err.to_string().contains("no pending interaction"),
            "got: {err}"
        );
    }

    // --- 10. resolve_wrong_type_returns_error ---

    #[tokio::test]
    async fn resolve_wrong_type_returns_error() {
        let (bridge, mut rx) = make_bridge(Duration::from_secs(5));
        let session_id = Uuid::new_v4();
        let on_input = bridge.make_on_input(session_id);

        // Start an input interaction
        let _handle = tokio::spawn(async move { on_input().await });

        // Get the interaction ID
        let msg = rx.recv().await.expect("should receive InputNeeded");
        let interaction_id = match msg {
            OutboundMessage::InputNeeded { interaction_id, .. } => interaction_id,
            other => panic!("expected InputNeeded, got: {other:?}"),
        };

        // Try resolving as approval (wrong type)
        let err = bridge
            .resolve_approval(interaction_id, ApprovalDecision::Allow)
            .expect_err("should error on wrong type");
        assert!(
            err.to_string().contains("not an approval interaction"),
            "got: {err}"
        );
    }

    // --- 11. concurrent_interactions ---

    #[tokio::test]
    async fn concurrent_interactions() {
        let (bridge, mut rx) = make_bridge(Duration::from_secs(5));
        let session_id = Uuid::new_v4();

        // Create three different interaction types
        let on_input = bridge.make_on_input(session_id);
        let on_question = bridge.make_on_question(session_id);
        let on_approval = bridge.make_on_approval(session_id);

        // Spawn all three
        let input_handle = tokio::spawn(async move { on_input().await });
        let question_handle = {
            let req = make_question_request();
            tokio::spawn(async move { on_question(req).await })
        };
        let approval_handle = tokio::task::spawn_blocking(move || on_approval(&[]));

        // Collect all three outbound messages (order not guaranteed)
        let mut input_id = None;
        let mut question_id = None;
        let mut approval_id = None;

        for _ in 0..3 {
            let msg = rx.recv().await.expect("should receive outbound message");
            match msg {
                OutboundMessage::InputNeeded { interaction_id, .. } => {
                    input_id = Some(interaction_id)
                }
                OutboundMessage::QuestionNeeded { interaction_id, .. } => {
                    question_id = Some(interaction_id)
                }
                OutboundMessage::ApprovalNeeded { interaction_id, .. } => {
                    approval_id = Some(interaction_id)
                }
                other => panic!("unexpected outbound message: {other:?}"),
            }
        }

        let input_id = input_id.expect("should have received InputNeeded");
        let question_id = question_id.expect("should have received QuestionNeeded");
        let approval_id = approval_id.expect("should have received ApprovalNeeded");

        // Resolve in reverse order (question, approval, input)
        bridge
            .resolve_question(
                question_id,
                QuestionResponse {
                    answers: vec![vec!["Blue".into()]],
                },
            )
            .expect("resolve question");
        bridge
            .resolve_approval(approval_id, ApprovalDecision::AlwaysAllow)
            .expect("resolve approval");
        bridge
            .resolve_input(input_id, Some("concurrent input".into()))
            .expect("resolve input");

        // Verify all three
        let input_result = input_handle.await.expect("input task");
        assert_eq!(input_result, Some("concurrent input".into()));

        let question_result = question_handle.await.expect("question task");
        let resp = question_result.expect("question should be Ok");
        assert_eq!(resp.answers, vec![vec!["Blue".to_string()]]);

        let approval_result = approval_handle.await.expect("approval task");
        assert_eq!(approval_result, ApprovalDecision::AlwaysAllow);
    }

    // --- Additional edge case tests ---

    #[tokio::test]
    async fn text_delta_multiple_sends() {
        let (bridge, mut rx) = make_bridge(Duration::from_secs(5));
        let session_id = Uuid::new_v4();
        let on_text = bridge.make_on_text(session_id);

        on_text("chunk1");
        on_text("chunk2");
        on_text("chunk3");

        for expected in &["chunk1", "chunk2", "chunk3"] {
            let msg = rx.recv().await.expect("should receive message");
            match msg {
                OutboundMessage::TextDelta { text, .. } => assert_eq!(text, *expected),
                other => panic!("expected TextDelta, got: {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn input_resolve_with_none_ends_session() {
        let (bridge, mut rx) = make_bridge(Duration::from_secs(5));
        let session_id = Uuid::new_v4();
        let on_input = bridge.make_on_input(session_id);

        let handle = tokio::spawn(async move { on_input().await });

        let msg = rx.recv().await.expect("should receive InputNeeded");
        let interaction_id = match msg {
            OutboundMessage::InputNeeded { interaction_id, .. } => interaction_id,
            other => panic!("expected InputNeeded, got: {other:?}"),
        };

        // Resolve with None (end session)
        bridge
            .resolve_input(interaction_id, None)
            .expect("resolve should succeed");

        let result = handle.await.expect("task should complete");
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn approval_needed_includes_tool_calls_json() {
        let (bridge, mut rx) = make_bridge(Duration::from_secs(5));
        let session_id = Uuid::new_v4();
        let on_approval = bridge.make_on_approval(session_id);

        let tool_calls = vec![crate::llm::types::ToolCall {
            id: "call-1".into(),
            name: "bash".into(),
            input: serde_json::json!({"command": "ls"}),
        }];

        let tool_calls_for_thread = tool_calls.clone();
        let handle = tokio::task::spawn_blocking(move || on_approval(&tool_calls_for_thread));

        let msg = rx.recv().await.expect("should receive ApprovalNeeded");
        match &msg {
            OutboundMessage::ApprovalNeeded {
                tool_calls: tc_json,
                interaction_id,
                ..
            } => {
                // Verify tool calls are serialized correctly
                assert!(tc_json.is_array());
                assert_eq!(tc_json[0]["name"], "bash");
                assert_eq!(tc_json[0]["input"]["command"], "ls");

                // Resolve to unblock the thread
                bridge
                    .resolve_approval(*interaction_id, ApprovalDecision::Deny)
                    .expect("resolve");
            }
            other => panic!("expected ApprovalNeeded, got: {other:?}"),
        }

        let result = handle.await.expect("task should complete");
        assert_eq!(result, ApprovalDecision::Deny);
    }

    #[tokio::test]
    async fn question_needed_includes_request() {
        let (bridge, mut rx) = make_bridge(Duration::from_secs(5));
        let session_id = Uuid::new_v4();
        let on_question = bridge.make_on_question(session_id);

        let request = make_question_request();
        let handle = tokio::spawn(async move { on_question(request).await });

        let msg = rx.recv().await.expect("should receive QuestionNeeded");
        match &msg {
            OutboundMessage::QuestionNeeded {
                request,
                interaction_id,
                ..
            } => {
                assert_eq!(request.questions.len(), 1);
                assert_eq!(request.questions[0].question, "Pick a color");
                assert_eq!(request.questions[0].options.len(), 2);

                bridge
                    .resolve_question(
                        *interaction_id,
                        QuestionResponse {
                            answers: vec![vec!["Blue".into()]],
                        },
                    )
                    .expect("resolve");
            }
            other => panic!("expected QuestionNeeded, got: {other:?}"),
        }

        let result = handle.await.expect("task should complete");
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn question_channel_closed_returns_error() {
        let (bridge, mut rx) = make_bridge(Duration::from_secs(5));
        let session_id = Uuid::new_v4();
        let on_question = bridge.make_on_question(session_id);

        let request = make_question_request();
        let handle = tokio::spawn(async move { on_question(request).await });

        // Get interaction ID, then manually drop the sender to simulate channel close
        let msg = rx.recv().await.expect("should receive QuestionNeeded");
        let interaction_id = match msg {
            OutboundMessage::QuestionNeeded { interaction_id, .. } => interaction_id,
            other => panic!("expected QuestionNeeded, got: {other:?}"),
        };

        // Take the pending sender and drop it (simulates channel close)
        {
            let mut pending = bridge.pending.write().expect("lock");
            pending.remove(&interaction_id);
            // Sender is dropped here
        }

        let result = handle.await.expect("task should complete");
        let err = result.expect_err("should be Err");
        assert!(err.to_string().contains("channel closed"), "got: {err}");
    }

    #[tokio::test]
    async fn resolve_approval_after_timeout_grace_period() {
        let (bridge, mut rx) = make_bridge(Duration::from_millis(10));
        let session_id = Uuid::new_v4();
        let on_approval = bridge.make_on_approval(session_id);

        let handle = tokio::task::spawn_blocking(move || on_approval(&[]));

        let msg = rx.recv().await.expect("should receive ApprovalNeeded");
        let interaction_id = match msg {
            OutboundMessage::ApprovalNeeded { interaction_id, .. } => interaction_id,
            other => panic!("expected ApprovalNeeded, got: {other:?}"),
        };

        // Wait for timeout
        let result = handle.await.expect("task should complete");
        assert_eq!(result, ApprovalDecision::Deny);

        // Late resolve during grace period should succeed (entry still present).
        bridge
            .resolve_approval(interaction_id, ApprovalDecision::Allow)
            .expect("late resolve during grace period should succeed");

        // Second resolve should fail (entry consumed by first resolve)
        let err = bridge
            .resolve_approval(interaction_id, ApprovalDecision::Allow)
            .expect_err("should error after entry consumed");
        assert!(
            err.to_string().contains("no pending interaction"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn resolve_question_after_timeout_grace_period() {
        let (bridge, mut rx) = make_bridge(Duration::from_millis(10));
        let session_id = Uuid::new_v4();
        let on_question = bridge.make_on_question(session_id);

        let request = make_question_request();
        let handle = tokio::spawn(async move { on_question(request).await });

        let msg = rx.recv().await.expect("should receive QuestionNeeded");
        let interaction_id = match msg {
            OutboundMessage::QuestionNeeded { interaction_id, .. } => interaction_id,
            other => panic!("expected QuestionNeeded, got: {other:?}"),
        };

        // Wait for timeout
        let result = handle.await.expect("task should complete");
        let err = result.expect_err("should be Err");
        assert!(err.to_string().contains("timed out"), "got: {err}");

        // Late resolve during grace period should succeed (entry still present).
        bridge
            .resolve_question(
                interaction_id,
                QuestionResponse {
                    answers: vec![vec!["too late".into()]],
                },
            )
            .expect("late resolve during grace period should succeed");

        // Second resolve should fail (entry consumed by first resolve)
        let err = bridge
            .resolve_question(
                interaction_id,
                QuestionResponse {
                    answers: vec![vec!["really late".into()]],
                },
            )
            .expect_err("should error after entry consumed");
        assert!(
            err.to_string().contains("no pending interaction"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn resolve_input_after_timeout_grace_period() {
        let (bridge, mut rx) = make_bridge(Duration::from_millis(10));
        let session_id = Uuid::new_v4();
        let on_input = bridge.make_on_input(session_id);

        let handle = tokio::spawn(async move { on_input().await });

        let msg = rx.recv().await.expect("should receive InputNeeded");
        let interaction_id = match msg {
            OutboundMessage::InputNeeded { interaction_id, .. } => interaction_id,
            other => panic!("expected InputNeeded, got: {other:?}"),
        };

        // Wait for timeout
        let result = handle.await.expect("task should complete");
        assert_eq!(result, None);

        // Late resolve during grace period should succeed (entry still present).
        // The oneshot receiver is already dropped, so the send is a no-op, but
        // the resolve call itself returns Ok.
        bridge
            .resolve_input(interaction_id, Some("too late".into()))
            .expect("late resolve during grace period should succeed");

        // Second resolve should fail (entry consumed by first resolve)
        let err = bridge
            .resolve_input(interaction_id, Some("really late".into()))
            .expect_err("should error after entry consumed");
        assert!(
            err.to_string().contains("no pending interaction"),
            "got: {err}"
        );
    }
}
