use std::sync::{Arc, Mutex};

use slint::Weak;

use heartbit::{
    AgentEvent, ApprovalDecision, Blackboard, OnApproval, OnEvent, OnInput, OnQuestion, OnText,
    QuestionRequest, QuestionResponse, ToolCall,
};

use crate::bridge::state::{EventProcessorState, PendingQuestion, PendingQuestionItem};
use crate::MainWindow;

/// Shared state between the tokio thread and the UI thread.
pub struct SharedState {
    pub processor: Mutex<EventProcessorState>,
    pub approval_tx: Mutex<Option<tokio::sync::oneshot::Sender<ApprovalDecision>>>,
    pub input_tx: Mutex<Option<tokio::sync::oneshot::Sender<Option<String>>>>,
    pub blackboard: Mutex<Option<Arc<dyn Blackboard>>>,
    pub question_tx: Mutex<Option<tokio::sync::oneshot::Sender<QuestionResponse>>>,
    pub pending_question: Mutex<Option<PendingQuestion>>,
}

impl SharedState {
    pub fn new() -> Self {
        Self {
            processor: Mutex::new(EventProcessorState::new()),
            approval_tx: Mutex::new(None),
            input_tx: Mutex::new(None),
            blackboard: Mutex::new(None),
            question_tx: Mutex::new(None),
            pending_question: Mutex::new(None),
        }
    }
}

/// Build the `OnEvent` callback that processes events and pushes state to the UI.
pub fn build_on_event(shared: Arc<SharedState>, ui_handle: Weak<MainWindow>) -> Arc<OnEvent> {
    Arc::new(move |event: AgentEvent| {
        // 1. Update shared state
        {
            let mut proc = shared.processor.lock().expect("processor lock poisoned");
            proc.process_event(&event);
        }

        // 2. Snapshot blackboard (if available) — runs on tokio worker thread
        snapshot_blackboard(&shared);

        // 3. Push to UI via event loop
        let shared_clone = Arc::clone(&shared);
        ui_handle
            .upgrade_in_event_loop(move |ui| {
                sync_state_to_ui(&shared_clone, &ui);
            })
            .ok();
    })
}

/// Build the `OnText` callback that streams text deltas to the current message.
pub fn build_on_text(shared: Arc<SharedState>, ui_handle: Weak<MainWindow>) -> Arc<OnText> {
    Arc::new(move |text: &str| {
        {
            let mut proc = shared.processor.lock().expect("processor lock poisoned");
            let agent = proc.current_agent().to_string();
            proc.update_streaming_text(&agent, text);
        }
        let shared_clone = Arc::clone(&shared);
        ui_handle
            .upgrade_in_event_loop(move |ui| {
                ui.set_run_status("running".into());
                sync_state_to_ui(&shared_clone, &ui);
            })
            .ok();
    })
}

/// Build the `OnApproval` callback using oneshot channels.
///
/// When approval is requested, the callback stores a oneshot sender in shared state,
/// signals the UI to show the approval banner, and blocks until the user responds.
pub fn build_on_approval(shared: Arc<SharedState>) -> Arc<OnApproval> {
    // Note: The UI is already updated via the ApprovalRequested event → on_event →
    // sync_state_to_ui pipeline. This callback only needs to store the oneshot sender
    // and block until the user responds.
    Arc::new(
        move |_tool_calls: &[ToolCall]| -> heartbit::ApprovalDecision {
            let (tx, rx) = tokio::sync::oneshot::channel();

            // Store the sender
            {
                let mut approval = shared
                    .approval_tx
                    .lock()
                    .expect("approval_tx lock poisoned");
                *approval = Some(tx);
            }

            // Block until user responds — use block_in_place since this runs on a tokio worker.
            // Channel now carries ApprovalDecision directly (supports 4 choices from UI).
            tokio::task::block_in_place(|| rx.blocking_recv().unwrap_or(ApprovalDecision::Deny))
        },
    )
}

/// Build the `OnInput` callback using oneshot channels.
///
/// When input is requested, signals the UI to enable the input field and
/// waits for the user to submit a message. Records user messages in the chat.
pub fn build_on_input(shared: Arc<SharedState>, ui_handle: Weak<MainWindow>) -> Arc<OnInput> {
    Arc::new(move || {
        let shared_clone = Arc::clone(&shared);
        let ui_handle_clone = ui_handle.clone();
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();

            // Store the sender
            {
                let mut input = shared_clone
                    .input_tx
                    .lock()
                    .expect("input_tx lock poisoned");
                *input = Some(tx);
            }

            // Set agent to idle and signal UI to enable input
            {
                let mut proc = shared_clone
                    .processor
                    .lock()
                    .expect("processor lock poisoned");
                proc.set_current_agent_idle();
            }
            let shared_ui = Arc::clone(&shared_clone);
            ui_handle_clone
                .upgrade_in_event_loop(move |ui| {
                    ui.set_input_requested(true);
                    sync_state_to_ui(&shared_ui, &ui);
                    ui.invoke_focus_input();
                })
                .ok();

            // Wait for user input
            let result = rx.await.unwrap_or(None);

            // Record user message in chat and reset input_requested
            let has_input = matches!(&result, Some(text) if !text.is_empty());
            if has_input {
                let mut proc = shared_clone
                    .processor
                    .lock()
                    .expect("processor lock poisoned");
                proc.push_user_message(result.as_deref().unwrap_or_default());
            }

            let shared_resume = Arc::clone(&shared_clone);
            ui_handle_clone
                .upgrade_in_event_loop(move |ui| {
                    ui.set_input_requested(false);
                    if has_input {
                        ui.set_run_status("running".into());
                    }
                    sync_state_to_ui(&shared_resume, &ui);
                })
                .ok();

            result
        })
    })
}

/// Build the `OnQuestion` callback using oneshot channels.
///
/// When the agent asks a structured question, the callback stores the question
/// state and a oneshot sender in shared state, signals the UI to show the
/// question dialog, and awaits the user's response.
pub fn build_on_question(shared: Arc<SharedState>, ui_handle: Weak<MainWindow>) -> Arc<OnQuestion> {
    Arc::new(move |request: QuestionRequest| {
        let shared_clone = Arc::clone(&shared);
        let ui_handle_clone = ui_handle.clone();
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();

            // Convert to PendingQuestion
            let items: Vec<PendingQuestionItem> = request
                .questions
                .iter()
                .map(|q| {
                    let options: Vec<(String, String)> = q
                        .options
                        .iter()
                        .map(|o| (o.label.clone(), o.description.clone()))
                        .collect();
                    let selected = vec![false; options.len()];
                    PendingQuestionItem {
                        header: q.header.clone(),
                        question: q.question.clone(),
                        options,
                        multiple: q.multiple,
                        selected,
                    }
                })
                .collect();
            let pending = PendingQuestion::new(items);

            // Store in shared state
            {
                let mut q_tx = shared_clone
                    .question_tx
                    .lock()
                    .expect("question_tx lock poisoned");
                *q_tx = Some(tx);
            }
            {
                let mut pq = shared_clone
                    .pending_question
                    .lock()
                    .expect("pending_question lock poisoned");
                *pq = Some(pending);
            }

            // Push current question to UI
            let shared_ui = Arc::clone(&shared_clone);
            ui_handle_clone
                .upgrade_in_event_loop(move |ui| {
                    sync_question_to_ui(&shared_ui, &ui);
                })
                .ok();

            // Wait for user response
            let response = rx
                .await
                .map_err(|_| heartbit::Error::Agent("Question dialog was cancelled".into()))?;
            Ok(response)
        })
    })
}

/// Synchronize the pending question state to the Slint UI.
pub fn sync_question_to_ui(shared: &SharedState, ui: &MainWindow) {
    let pq = shared
        .pending_question
        .lock()
        .expect("pending_question lock poisoned");
    match pq.as_ref().and_then(|p| p.current()) {
        Some(q) => {
            let progress = pq.as_ref().expect("just checked").progress_text();
            let options_model: Vec<crate::QuestionOptionData> = q
                .options
                .iter()
                .enumerate()
                .map(|(i, (label, desc))| crate::QuestionOptionData {
                    label: label.clone().into(),
                    description: desc.clone().into(),
                    selected: q.selected.get(i).copied().unwrap_or(false),
                    index: i as i32,
                })
                .collect();
            ui.set_question_pending(true);
            ui.set_question_header(q.header.clone().into());
            ui.set_question_text(q.question.clone().into());
            ui.set_question_multiple(q.multiple);
            ui.set_question_progress(progress.into());
            let opts_rc = std::rc::Rc::new(slint::VecModel::from(options_model));
            ui.set_question_options(opts_rc.into());
        }
        None => {
            ui.set_question_pending(false);
        }
    }
}

/// Snapshot the blackboard into the processor state.
fn snapshot_blackboard(shared: &SharedState) {
    // Clone the Arc and drop the mutex guard before blocking
    let bb_clone = {
        let bb = shared.blackboard.lock().expect("blackboard lock poisoned");
        match *bb {
            Some(ref blackboard) => Some(Arc::clone(blackboard)),
            None => None,
        }
    };
    if let Some(blackboard) = bb_clone {
        // OnEvent runs on a tokio worker thread, so we can block_on
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let entries = tokio::task::block_in_place(|| {
                handle.block_on(async {
                    let keys = match blackboard.list_keys().await {
                        Ok(k) => k,
                        Err(_) => return Vec::new(),
                    };
                    let mut result = Vec::with_capacity(keys.len());
                    for key in keys {
                        if let Ok(Some(val)) = blackboard.read(&key).await {
                            let display = match val {
                                serde_json::Value::String(s) => s,
                                other => other.to_string(),
                            };
                            result.push((key, display));
                        }
                    }
                    result
                })
            });
            let mut proc = shared.processor.lock().expect("processor lock poisoned");
            proc.set_blackboard_entries(entries);
        }
    }
}

/// Compute turn budget fraction (0.0 to 1.0).
fn turn_fraction(current: usize, max: usize) -> f32 {
    if max > 0 {
        current as f32 / max as f32
    } else {
        0.0
    }
}

/// Format a latency value for human-readable display.
///
/// - 0: "" (empty — hidden in UI)
/// - < 1000ms: "123ms"
/// - >= 1000ms: "1.2s"
fn format_latency(ms: u64) -> String {
    if ms == 0 {
        String::new()
    } else if ms < 1_000 {
        format!("{ms}ms")
    } else {
        format!("{:.1}s", ms as f64 / 1_000.0)
    }
}

/// Format a token count for human-readable display.
///
/// - < 1000: "123"
/// - 1000..999_999: "12.3k"
/// - >= 1_000_000: "1.2M"
fn format_tokens(n: u32) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        let k = n as f64 / 1_000.0;
        if n < 10_000 {
            format!("{k:.1}k")
        } else {
            format!("{:.0}k", k)
        }
    } else {
        let m = n as f64 / 1_000_000.0;
        format!("{m:.1}M")
    }
}

/// Synchronize the `EventProcessorState` snapshot to Slint UI models.
pub fn sync_state_to_ui(shared: &SharedState, ui: &MainWindow) {
    let proc = shared.processor.lock().expect("processor lock poisoned");

    // Update agents (use visible_agents for history support)
    let filter = proc.agent_filter();
    let agents_model: Vec<crate::AgentState> = {
        let mut agents: Vec<_> = proc.visible_agents().values().collect();
        agents.sort_by(|a, b| a.name.cmp(&b.name));
        agents
            .iter()
            .map(|a| crate::AgentState {
                name: a.name.clone().into(),
                status: a.status.as_str().into(),
                current_turn: a.current_turn as i32,
                max_turns: a.max_turns as i32,
                tool_name: a.current_tool.clone().unwrap_or_default().into(),
                tokens_in: format_tokens(a.tokens_in).into(),
                tokens_out: format_tokens(a.tokens_out).into(),
                turn_fraction: turn_fraction(a.current_turn, a.max_turns),
                selected: filter == Some(a.name.as_str()),
            })
            .collect()
    };
    let agents_rc = std::rc::Rc::new(slint::VecModel::from(agents_model));
    ui.set_agents(agents_rc.into());

    // Update messages (preserving expansion state)
    // Use visible_messages/visible_stats for history support
    // Apply agent filter: user messages always shown, others filtered by agent name
    let messages_model: Vec<crate::ChatMsg> = proc
        .visible_messages()
        .iter()
        .filter(|m| {
            use crate::bridge::state::MessageKind;
            match filter {
                None => true,
                Some(agent_name) => match m.kind {
                    // Always show user messages (they provide task context)
                    MessageKind::User => true,
                    // Show system/text/tool messages only from the filtered agent
                    _ => m.agent == agent_name,
                },
            }
        })
        .map(|m| {
            use crate::bridge::state::MessageKind;
            let role: slint::SharedString = match m.kind {
                MessageKind::Text => "assistant",
                MessageKind::ToolCall => "tool",
                MessageKind::System => "system",
                MessageKind::User => "user",
            }
            .into();
            crate::ChatMsg {
                id: m.id,
                agent: m.agent.clone().into(),
                role,
                content: m.content.clone().into(),
                is_tool_call: m.kind == MessageKind::ToolCall,
                tool_name: m.tool_name.clone().unwrap_or_default().into(),
                tool_input: m.tool_input.clone().unwrap_or_default().into(),
                tool_output: m.tool_output.clone().unwrap_or_default().into(),
                tool_is_error: m.tool_is_error,
                tool_duration: format_latency(m.tool_duration_ms).into(),
                is_expanded: proc.is_expanded(m.id),
            }
        })
        .collect();
    let new_count = messages_model.len();
    let messages_rc = std::rc::Rc::new(slint::VecModel::from(messages_model));
    ui.set_messages(messages_rc.into());

    // Auto-scroll chat to bottom when content changes
    // (message count changed or streaming text updated existing message)
    if new_count > 0 {
        ui.invoke_scroll_chat_to_bottom();
    }

    // Update stats (use visible_stats for history support)
    let stats = proc.visible_stats();
    let cost_str = format!("${:.4}", stats.estimated_cost);
    let elapsed_str = if proc.viewing_history().is_none() {
        let elapsed_secs = proc.elapsed_seconds();
        if elapsed_secs > 0 {
            format!("{}:{:02}", elapsed_secs / 60, elapsed_secs % 60)
        } else {
            String::new()
        }
    } else {
        stats.frozen_elapsed_string()
    };
    ui.set_stats(crate::TokenStats {
        total_input: format_tokens(stats.total_input).into(),
        total_output: format_tokens(stats.total_output).into(),
        reasoning: format_tokens(stats.reasoning).into(),
        cache_created: format_tokens(stats.cache_created).into(),
        cache_read: format_tokens(stats.cache_read).into(),
        total_cost: cost_str.into(),
        tool_calls: stats.tool_calls.to_string().into(),
        last_latency: format_latency(stats.last_latency_ms).into(),
        model: stats.model.clone().unwrap_or_default().into(),
        elapsed: elapsed_str.into(),
    });

    // Update approval state
    match proc.pending_approval() {
        Some(approval) => {
            ui.set_approval_pending(true);
            ui.set_approval_tool_names(approval.tool_names.join(", ").into());
            ui.set_approval_agent_name(approval.agent.clone().into());
        }
        None => {
            ui.set_approval_pending(false);
        }
    }

    // Update blackboard entries (only for live view — historical blackboard is not preserved)
    let bb_model: Vec<crate::BlackboardData> = if proc.viewing_history().is_none() {
        proc.blackboard_entries()
            .iter()
            .map(|(k, v)| crate::BlackboardData {
                key: k.clone().into(),
                value: v.clone().into(),
            })
            .collect()
    } else {
        Vec::new()
    };
    let bb_rc = std::rc::Rc::new(slint::VecModel::from(bb_model));
    ui.set_blackboard_entries(bb_rc.into());

    // Update history entries
    let history_model: Vec<crate::HistoryEntry> = proc
        .history()
        .iter()
        .map(|r| crate::HistoryEntry {
            id: r.id as i32,
            task: r.task.clone().into(),
            status: r.status.into(),
            cost: r.cost.clone().into(),
            elapsed: r.elapsed.clone().into(),
            msg_count: r.message_count as i32,
        })
        .collect();
    let history_rc = std::rc::Rc::new(slint::VecModel::from(history_model));
    ui.set_history_entries(history_rc.into());
    ui.set_viewing_history_id(proc.viewing_history().map_or(-1, |id| id as i32));
    ui.set_agent_filter(filter.unwrap_or("").into());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_tokens_small() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(1), "1");
        assert_eq!(format_tokens(999), "999");
    }

    #[test]
    fn format_tokens_thousands() {
        assert_eq!(format_tokens(1_000), "1.0k");
        assert_eq!(format_tokens(1_234), "1.2k");
        assert_eq!(format_tokens(9_999), "10.0k");
        assert_eq!(format_tokens(10_000), "10k");
        assert_eq!(format_tokens(99_500), "100k");
        assert_eq!(format_tokens(123_456), "123k");
    }

    #[test]
    fn format_tokens_millions() {
        assert_eq!(format_tokens(1_000_000), "1.0M");
        assert_eq!(format_tokens(1_234_567), "1.2M");
        assert_eq!(format_tokens(12_345_678), "12.3M");
    }

    #[test]
    fn format_latency_zero() {
        assert_eq!(format_latency(0), "");
    }

    #[test]
    fn format_latency_milliseconds() {
        assert_eq!(format_latency(1), "1ms");
        assert_eq!(format_latency(123), "123ms");
        assert_eq!(format_latency(999), "999ms");
    }

    #[test]
    fn format_latency_seconds() {
        assert_eq!(format_latency(1_000), "1.0s");
        assert_eq!(format_latency(1_500), "1.5s");
        assert_eq!(format_latency(12_345), "12.3s");
    }

    #[test]
    fn turn_fraction_zero_max() {
        assert_eq!(turn_fraction(0, 0), 0.0);
        assert_eq!(turn_fraction(5, 0), 0.0);
    }

    #[test]
    fn turn_fraction_normal() {
        assert!((turn_fraction(1, 10) - 0.1).abs() < f32::EPSILON);
        assert!((turn_fraction(5, 10) - 0.5).abs() < f32::EPSILON);
        assert!((turn_fraction(10, 10) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn turn_fraction_partial() {
        let f = turn_fraction(3, 200);
        assert!(f > 0.0 && f < 0.1);
    }
}
