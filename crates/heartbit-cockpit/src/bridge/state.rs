use std::collections::{HashMap, HashSet};

use heartbit::{AgentEvent, TokenUsage};

/// Try to pretty-print a JSON string. Falls back to the original if parsing fails.
fn pretty_print_json(s: &str) -> String {
    serde_json::from_str::<serde_json::Value>(s)
        .ok()
        .and_then(|v| serde_json::to_string_pretty(&v).ok())
        .unwrap_or_else(|| s.to_string())
}

/// Agent status in the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStatus {
    Idle,
    Thinking,
    Executing,
    Completed,
    Failed,
    Cancelled,
}

impl AgentStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Thinking => "thinking",
            Self::Executing => "executing",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// UI-ready state for a single agent.
#[derive(Debug, Clone)]
pub struct AgentUiState {
    pub name: String,
    pub status: AgentStatus,
    pub current_turn: usize,
    pub max_turns: usize,
    pub current_tool: Option<String>,
    pub tokens_in: u32,
    pub tokens_out: u32,
}

/// Kind of chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageKind {
    Text,
    ToolCall,
    System,
    User,
}

/// A single chat message for the UI.
#[derive(Debug, Clone)]
pub struct ChatMessageData {
    pub id: i32,
    pub agent: String,
    pub kind: MessageKind,
    pub content: String,
    pub tool_name: Option<String>,
    pub tool_input: Option<String>,
    pub tool_output: Option<String>,
    pub tool_is_error: bool,
    pub tool_duration_ms: u64,
}

/// Accumulated statistics.
#[derive(Debug, Clone, Default)]
pub struct StatsData {
    pub total_input: u32,
    pub total_output: u32,
    pub cache_created: u32,
    pub cache_read: u32,
    pub reasoning: u32,
    pub estimated_cost: f64,
    pub tool_calls: usize,
    pub last_latency_ms: u64,
    pub model: Option<String>,
    /// Live timer start — `None` when not running or after freeze.
    run_start: Option<std::time::Instant>,
    /// Frozen elapsed seconds — set when run ends, displayed in stats.
    frozen_elapsed_secs: u64,
}

impl StatsData {
    /// Format the frozen elapsed time for display.
    pub fn frozen_elapsed_string(&self) -> String {
        let secs = self.frozen_elapsed_secs;
        if secs > 0 {
            format!("{}:{:02}", secs / 60, secs % 60)
        } else {
            String::new()
        }
    }
}

/// Approval request data.
#[derive(Debug, Clone)]
pub struct ApprovalData {
    pub agent: String,
    pub tool_names: Vec<String>,
}

/// An archived run for the history panel.
#[derive(Debug, Clone)]
pub struct ArchivedRun {
    pub id: usize,
    pub task: String,
    pub status: &'static str,
    pub cost: String,
    pub elapsed: String,
    pub message_count: usize,
    pub messages: Vec<ChatMessageData>,
    pub stats: StatsData,
    pub agents: HashMap<String, AgentUiState>,
}

/// Send+Sync state machine that processes `AgentEvent` values and
/// accumulates UI-ready state. No Slint types — just owned data.
#[derive(Debug)]
pub struct EventProcessorState {
    agents: HashMap<String, AgentUiState>,
    messages: Vec<ChatMessageData>,
    pending_approval: Option<ApprovalData>,
    stats: StatsData,
    next_message_id: i32,
    /// Maps tool_call_id → message ID for updating tool output later.
    tool_call_message_index: HashMap<String, i32>,
    /// Tracks which message IDs have their tool output expanded.
    expanded_messages: HashSet<i32>,
    /// Active streaming message ID (replaced when LlmResponse arrives).
    streaming_message_id: Option<i32>,
    /// Buffer for accumulated streaming text deltas.
    streaming_buffer: String,
    /// Blackboard key-value snapshot for UI display.
    blackboard_entries: Vec<(String, String)>,
    /// Name of the most recently started agent (for streaming text attribution).
    current_agent: String,
    /// Archived past runs.
    history: Vec<ArchivedRun>,
    /// ID for the currently viewed historical run, or None for live view.
    viewing_history: Option<usize>,
    /// Agent name filter for chat messages (None = show all).
    agent_filter: Option<String>,
}

impl EventProcessorState {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            messages: Vec::new(),
            pending_approval: None,
            stats: StatsData::default(),
            next_message_id: 0,
            tool_call_message_index: HashMap::new(),
            expanded_messages: HashSet::new(),
            streaming_message_id: None,
            streaming_buffer: String::new(),
            blackboard_entries: Vec::new(),
            current_agent: "heartbit".to_string(),
            history: Vec::new(),
            viewing_history: None,
            agent_filter: None,
        }
    }

    #[cfg(test)]
    pub fn agents(&self) -> &HashMap<String, AgentUiState> {
        &self.agents
    }

    #[cfg(test)]
    pub fn messages(&self) -> &[ChatMessageData] {
        &self.messages
    }

    pub fn pending_approval(&self) -> Option<&ApprovalData> {
        self.pending_approval.as_ref()
    }

    #[cfg(test)]
    pub fn stats(&self) -> &StatsData {
        &self.stats
    }

    /// Elapsed seconds since the run started (live or frozen).
    pub fn elapsed_seconds(&self) -> u64 {
        match self.stats.run_start {
            Some(t) => t.elapsed().as_secs(),
            None => self.stats.frozen_elapsed_secs,
        }
    }

    /// Freeze the elapsed timer at the current value (call when run ends).
    pub fn freeze_elapsed(&mut self) {
        if let Some(start) = self.stats.run_start.take() {
            self.stats.frozen_elapsed_secs = start.elapsed().as_secs();
        }
    }

    pub fn blackboard_entries(&self) -> &[(String, String)] {
        &self.blackboard_entries
    }

    /// Name of the most recently started agent (for streaming attribution).
    pub fn current_agent(&self) -> &str {
        &self.current_agent
    }

    /// Set the current agent to idle (e.g., when awaiting user input).
    pub fn set_current_agent_idle(&mut self) {
        if let Some(a) = self.agents.get_mut(&self.current_agent) {
            a.status = AgentStatus::Idle;
        }
    }

    /// Archive the current run (if it has messages) and reset for a fresh run.
    pub fn reset(&mut self) {
        // Archive the current run if it has meaningful content (more than just user message)
        if self.messages.len() > 1 {
            let task = self
                .messages
                .iter()
                .find(|m| m.kind == MessageKind::User)
                .map(|m| m.content.clone())
                .unwrap_or_default();
            let status = self.final_status();
            let cost = format!("${:.4}", self.stats.estimated_cost);
            let elapsed = self.format_elapsed();
            let id = self.history.len();
            self.history.push(ArchivedRun {
                id,
                task,
                status,
                cost,
                elapsed,
                message_count: self.messages.len(),
                messages: self.messages.clone(),
                stats: self.stats.clone(),
                agents: self.agents.clone(),
            });
        }
        self.agents.clear();
        self.messages.clear();
        self.pending_approval = None;
        self.stats = StatsData::default();
        self.next_message_id = 0;
        self.tool_call_message_index.clear();
        self.expanded_messages.clear();
        self.streaming_message_id = None;
        self.streaming_buffer.clear();
        self.blackboard_entries.clear();
        self.current_agent = "heartbit".to_string();
        self.viewing_history = None;
        self.agent_filter = None;
    }

    /// Determine the final status label based on agent states.
    fn final_status(&self) -> &'static str {
        let has_failed = self
            .agents
            .values()
            .any(|a| a.status == AgentStatus::Failed);
        let has_cancelled = self
            .agents
            .values()
            .any(|a| a.status == AgentStatus::Cancelled);
        if has_failed {
            "failed"
        } else if has_cancelled {
            "cancelled"
        } else {
            "completed"
        }
    }

    /// Format elapsed time for archival.
    fn format_elapsed(&self) -> String {
        let secs = self.elapsed_seconds();
        if secs > 0 {
            format!("{}:{:02}", secs / 60, secs % 60)
        } else {
            String::new()
        }
    }

    /// Access the run history.
    pub fn history(&self) -> &[ArchivedRun] {
        &self.history
    }

    /// Set which historical run to view (None for live view).
    ///
    /// Clears expansion state because message IDs restart at 0 each run,
    /// so expansions from one view would incorrectly bleed into another.
    pub fn view_history(&mut self, id: Option<usize>) {
        self.viewing_history = id;
        self.expanded_messages.clear();
    }

    /// Get the currently viewed historical run ID (None = live view).
    pub fn viewing_history(&self) -> Option<usize> {
        self.viewing_history
    }

    /// Get agents for the current view (live or historical).
    pub fn visible_agents(&self) -> &HashMap<String, AgentUiState> {
        match self.viewing_history {
            Some(id) => self
                .history
                .get(id)
                .map(|r| &r.agents)
                .unwrap_or(&self.agents),
            None => &self.agents,
        }
    }

    /// Get messages for the current view (live or historical).
    pub fn visible_messages(&self) -> &[ChatMessageData] {
        match self.viewing_history {
            Some(id) => self
                .history
                .get(id)
                .map(|r| r.messages.as_slice())
                .unwrap_or(&[]),
            None => &self.messages,
        }
    }

    /// Get stats for the current view (live or historical).
    pub fn visible_stats(&self) -> &StatsData {
        match self.viewing_history {
            Some(id) => self
                .history
                .get(id)
                .map(|r| &r.stats)
                .unwrap_or(&self.stats),
            None => &self.stats,
        }
    }

    /// Toggle the agent filter. If the same agent is clicked again, clear the filter.
    pub fn toggle_agent_filter(&mut self, agent: &str) {
        if self.agent_filter.as_deref() == Some(agent) {
            self.agent_filter = None;
        } else {
            self.agent_filter = Some(agent.to_string());
        }
    }

    /// Get the current agent filter name (None = show all).
    pub fn agent_filter(&self) -> Option<&str> {
        self.agent_filter.as_deref()
    }

    pub fn set_blackboard_entries(&mut self, entries: Vec<(String, String)>) {
        self.blackboard_entries = entries;
    }

    /// Toggle expansion state for a message; returns the new state.
    pub fn toggle_expanded(&mut self, msg_id: i32) -> bool {
        if !self.expanded_messages.remove(&msg_id) {
            self.expanded_messages.insert(msg_id);
            true
        } else {
            false
        }
    }

    /// Check whether a message is expanded.
    pub fn is_expanded(&self, msg_id: i32) -> bool {
        self.expanded_messages.contains(&msg_id)
    }

    /// Append a streaming text delta, creating or updating the streaming message.
    pub fn update_streaming_text(&mut self, agent: &str, delta: &str) {
        self.streaming_buffer.push_str(delta);
        match self.streaming_message_id {
            Some(id) => {
                // Update existing streaming message in-place
                if let Some(msg) = self.messages.iter_mut().find(|m| m.id == id) {
                    msg.content.clone_from(&self.streaming_buffer);
                }
            }
            None => {
                let id = self.next_id();
                self.messages.push(ChatMessageData {
                    id,
                    agent: agent.to_string(),
                    kind: MessageKind::Text,
                    content: self.streaming_buffer.clone(),
                    tool_name: None,
                    tool_input: None,
                    tool_output: None,
                    tool_is_error: false,
                    tool_duration_ms: 0,
                });
                self.streaming_message_id = Some(id);
            }
        }
    }

    /// Finalize streaming: keep the full-text message in place and clear streaming state.
    ///
    /// Returns `true` if a streaming message was active and has been finalized.
    /// The message content is preserved from the buffer (which has the full,
    /// un-truncated text), unlike the `LlmResponse` event text which is truncated
    /// to `EVENT_MAX_PAYLOAD_BYTES`.
    pub fn finalize_streaming(&mut self) -> bool {
        let had_streaming = self.streaming_message_id.take().is_some();
        self.streaming_buffer.clear();
        had_streaming
    }

    /// Clear any pending approval (e.g., after task cancellation).
    pub fn clear_pending_approval(&mut self) {
        self.pending_approval = None;
    }

    /// Mark all non-terminal agents as cancelled and finalize streaming state.
    pub fn cancel_active_agents(&mut self) {
        for agent in self.agents.values_mut() {
            match agent.status {
                AgentStatus::Thinking | AgentStatus::Executing | AgentStatus::Idle => {
                    agent.status = AgentStatus::Cancelled;
                    agent.current_tool = None;
                }
                AgentStatus::Completed | AgentStatus::Failed | AgentStatus::Cancelled => {}
            }
        }
        // Finalize any in-progress streaming message (append truncation marker)
        if let Some(id) = self.streaming_message_id.take() {
            if let Some(msg) = self.messages.iter_mut().find(|m| m.id == id) {
                if !msg.content.is_empty() {
                    msg.content.push_str(" [cancelled]");
                }
            }
        }
        self.streaming_buffer.clear();
    }

    /// Set the estimated cost (used by runtime after run completes).
    pub fn set_estimated_cost(&mut self, cost: f64) {
        self.stats.estimated_cost = cost;
    }

    /// Process a single agent event, updating internal state.
    pub fn process_event(&mut self, event: &AgentEvent) {
        match event {
            AgentEvent::RunStarted { agent, task } => {
                self.current_agent.clone_from(agent);
                let is_first = self.agents.is_empty();
                if self.stats.run_start.is_none() {
                    self.stats.run_start = Some(std::time::Instant::now());
                }
                self.agents.insert(
                    agent.clone(),
                    AgentUiState {
                        name: agent.clone(),
                        status: AgentStatus::Thinking,
                        current_turn: 0,
                        max_turns: 0,
                        current_tool: None,
                        tokens_in: 0,
                        tokens_out: 0,
                    },
                );
                // First agent: short message (user message already shows the task).
                // Sub-agents: include the delegated task text.
                if is_first {
                    self.push_system_message_for(agent, format!("Agent '{agent}' started"));
                } else {
                    self.push_system_message_for(agent, format!("Agent '{agent}': {task}"));
                }
            }

            AgentEvent::TurnStarted {
                agent,
                turn,
                max_turns,
            } => {
                self.current_agent.clone_from(agent);
                if let Some(a) = self.agents.get_mut(agent) {
                    a.current_turn = *turn;
                    a.max_turns = *max_turns;
                }
            }

            AgentEvent::LlmResponse {
                agent,
                usage,
                text,
                latency_ms,
                model,
                ..
            } => {
                if let Some(a) = self.agents.get_mut(agent) {
                    a.tokens_in += usage.input_tokens;
                    a.tokens_out += usage.output_tokens;
                }
                self.stats.last_latency_ms = *latency_ms;
                if let Some(m) = model {
                    self.stats.model = Some(m.clone());
                }
                self.accumulate_tokens(usage);

                // Finalize streaming: keep full-text message from streaming buffer
                // (LlmResponse.text is truncated to EVENT_MAX_PAYLOAD_BYTES)
                let had_streaming = self.finalize_streaming();

                // Only create a new message if we weren't streaming (e.g., non-streaming provider)
                if !had_streaming && !text.is_empty() {
                    let id = self.next_id();
                    self.messages.push(ChatMessageData {
                        id,
                        agent: agent.clone(),
                        kind: MessageKind::Text,
                        content: text.clone(),
                        tool_name: None,
                        tool_input: None,
                        tool_output: None,
                        tool_is_error: false,
                        tool_duration_ms: 0,
                    });
                }
            }

            AgentEvent::ToolCallStarted {
                agent,
                tool_name,
                tool_call_id,
                input,
            } => {
                if let Some(a) = self.agents.get_mut(agent) {
                    a.status = AgentStatus::Executing;
                    a.current_tool = Some(tool_name.clone());
                }
                self.stats.tool_calls += 1;
                let id = self.next_id();
                let formatted_input = pretty_print_json(input);
                self.messages.push(ChatMessageData {
                    id,
                    agent: agent.clone(),
                    kind: MessageKind::ToolCall,
                    content: format!("Calling {tool_name}"),
                    tool_name: Some(tool_name.clone()),
                    tool_input: Some(formatted_input),
                    tool_output: None,
                    tool_is_error: false,
                    tool_duration_ms: 0,
                });
                self.tool_call_message_index
                    .insert(tool_call_id.clone(), id);
            }

            AgentEvent::ToolCallCompleted {
                agent,
                tool_call_id,
                is_error,
                duration_ms,
                output,
                ..
            } => {
                if let Some(a) = self.agents.get_mut(agent) {
                    a.status = AgentStatus::Thinking;
                    a.current_tool = None;
                }
                if let Some(&msg_id) = self.tool_call_message_index.get(tool_call_id) {
                    if let Some(msg) = self.messages.iter_mut().find(|m| m.id == msg_id) {
                        msg.tool_output = Some(output.clone());
                        msg.tool_is_error = *is_error;
                        msg.tool_duration_ms = *duration_ms;
                    }
                }
            }

            AgentEvent::ApprovalRequested {
                agent, tool_names, ..
            } => {
                self.pending_approval = Some(ApprovalData {
                    agent: agent.clone(),
                    tool_names: tool_names.clone(),
                });
            }

            AgentEvent::ApprovalDecision { .. } => {
                self.pending_approval = None;
            }

            AgentEvent::SubAgentsDispatched { agents, .. } => {
                for name in agents {
                    self.agents.entry(name.clone()).or_insert(AgentUiState {
                        name: name.clone(),
                        status: AgentStatus::Idle,
                        current_turn: 0,
                        max_turns: 0,
                        current_tool: None,
                        tokens_in: 0,
                        tokens_out: 0,
                    });
                }
            }

            AgentEvent::SubAgentCompleted { agent, success, .. } => {
                // Only update status — don't accumulate tokens here because sub-agent
                // LlmResponse events already accumulated them (forwarded via on_event).
                if let Some(a) = self.agents.get_mut(agent) {
                    a.status = if *success {
                        AgentStatus::Completed
                    } else {
                        AgentStatus::Failed
                    };
                }
            }

            AgentEvent::ContextSummarized { agent, turn, usage } => {
                self.accumulate_tokens(usage);
                self.push_system_message_for(agent, format!("Context summarized at turn {turn}"));
            }

            AgentEvent::RunCompleted { agent, .. } => {
                // Only update status — don't overwrite global stats because:
                // 1. Per-turn LlmResponse events already accumulated tokens correctly
                // 2. Sub-agent RunCompleted would overwrite with sub-agent-only totals,
                //    losing orchestrator tokens and causing mid-run stat jumps
                if let Some(a) = self.agents.get_mut(agent) {
                    a.status = AgentStatus::Completed;
                }
            }

            AgentEvent::GuardrailDenied {
                agent,
                hook,
                reason,
                tool_name,
            } => {
                let detail = match tool_name {
                    Some(t) => format!("Guardrail ({hook}) denied tool '{t}': {reason}"),
                    None => format!("Guardrail ({hook}) denied: {reason}"),
                };
                self.push_system_message_for(agent, detail);
            }

            AgentEvent::RunFailed {
                agent,
                error,
                partial_usage,
            } => {
                if let Some(a) = self.agents.get_mut(agent) {
                    a.status = AgentStatus::Failed;
                }
                self.accumulate_tokens(partial_usage);
                self.push_system_message_for(agent, format!("Agent '{agent}' failed: {error}"));
            }

            // New observability events — display as system messages
            AgentEvent::RetryAttempt {
                agent,
                attempt,
                max_retries,
                delay_ms,
                error_class,
            } => {
                self.push_system_message_for(
                    agent,
                    format!("Retry {attempt}/{max_retries} ({error_class}), waiting {delay_ms}ms"),
                );
            }

            AgentEvent::DoomLoopDetected {
                agent,
                turn,
                consecutive_count,
                tool_names,
            } => {
                self.push_system_message_for(
                    agent,
                    format!(
                        "Doom loop detected at turn {turn}: {consecutive_count} identical calls ({})",
                        tool_names.join(", ")
                    ),
                );
            }

            AgentEvent::AutoCompactionTriggered {
                agent,
                turn,
                success,
                ..
            } => {
                let status = if *success { "succeeded" } else { "failed" };
                self.push_system_message_for(
                    agent,
                    format!("Auto-compaction {status} at turn {turn}"),
                );
            }

            AgentEvent::SessionPruned {
                agent,
                turn,
                tool_results_pruned,
                bytes_saved,
                ..
            } => {
                self.push_system_message_for(
                    agent,
                    format!(
                        "Session pruned at turn {turn}: {tool_results_pruned} tool results, {bytes_saved} bytes saved"
                    ),
                );
            }

            // Sensor events, routing events, and cascade events are daemon-only; no cockpit handling needed.
            AgentEvent::SensorEventProcessed { .. }
            | AgentEvent::StoryUpdated { .. }
            | AgentEvent::TaskRouted { .. }
            | AgentEvent::ModelEscalated { .. } => {}
        }
    }

    fn next_id(&mut self) -> i32 {
        let id = self.next_message_id;
        self.next_message_id += 1;
        id
    }

    /// Push a system message without agent attribution (e.g., cancellation).
    pub fn push_system_message(&mut self, content: &str) {
        self.push_system_message_for("", content.to_string());
    }

    fn push_system_message_for(&mut self, agent: &str, content: String) {
        let id = self.next_id();
        self.messages.push(ChatMessageData {
            id,
            agent: agent.to_string(),
            kind: MessageKind::System,
            content,
            tool_name: None,
            tool_input: None,
            tool_output: None,
            tool_is_error: false,
            tool_duration_ms: 0,
        });
    }

    /// Push a user message to the chat timeline.
    pub fn push_user_message(&mut self, content: &str) {
        let id = self.next_id();
        self.messages.push(ChatMessageData {
            id,
            agent: String::new(),
            kind: MessageKind::User,
            content: content.to_string(),
            tool_name: None,
            tool_input: None,
            tool_output: None,
            tool_is_error: false,
            tool_duration_ms: 0,
        });
    }

    /// Push a question-answered message showing the user's selection.
    pub fn push_question_answer(&mut self, header: &str, selected: &[String]) {
        let content = format!("{header}: {}", selected.join(", "));
        self.push_user_message(&content);
    }

    fn accumulate_tokens(&mut self, usage: &TokenUsage) {
        self.stats.total_input += usage.input_tokens;
        self.stats.total_output += usage.output_tokens;
        self.stats.cache_created += usage.cache_creation_input_tokens;
        self.stats.cache_read += usage.cache_read_input_tokens;
        self.stats.reasoning += usage.reasoning_tokens;
        self.update_running_cost();
    }

    fn update_running_cost(&mut self) {
        if let Some(ref model) = self.stats.model {
            let usage = TokenUsage {
                input_tokens: self.stats.total_input,
                output_tokens: self.stats.total_output,
                cache_creation_input_tokens: self.stats.cache_created,
                cache_read_input_tokens: self.stats.cache_read,
                reasoning_tokens: self.stats.reasoning,
            };
            if let Some(cost) = heartbit::estimate_cost(model, &usage) {
                self.stats.estimated_cost = cost;
            }
        }
    }
}

/// A single question in a pending question dialog.
#[derive(Debug, Clone)]
pub struct PendingQuestionItem {
    pub header: String,
    pub question: String,
    /// (label, description) pairs.
    pub options: Vec<(String, String)>,
    pub multiple: bool,
    /// Per-option selection state (used for multi-select toggle).
    pub selected: Vec<bool>,
}

/// State machine for structured agent-to-user questions.
///
/// Tracks the current question index, accumulated answers, and per-option
/// selection state for multi-select questions. Progression:
/// - Single-select: `select_option()` records answer and advances automatically.
/// - Multi-select: `select_option()` toggles, `submit_current()` records and advances.
#[derive(Debug)]
pub struct PendingQuestion {
    questions: Vec<PendingQuestionItem>,
    current_index: usize,
    answers: Vec<Vec<String>>,
}

impl PendingQuestion {
    pub fn new(items: Vec<PendingQuestionItem>) -> Self {
        let answers = Vec::with_capacity(items.len());
        Self {
            questions: items,
            current_index: 0,
            answers,
        }
    }

    /// Current question, if any remain.
    pub fn current(&self) -> Option<&PendingQuestionItem> {
        self.questions.get(self.current_index)
    }

    /// Progress text (e.g., "1/3"). Empty for single-question sets.
    pub fn progress_text(&self) -> String {
        if self.questions.len() <= 1 {
            String::new()
        } else {
            format!("{}/{}", self.current_index + 1, self.questions.len())
        }
    }

    /// Whether all questions have been answered.
    pub fn is_complete(&self) -> bool {
        self.answers.len() >= self.questions.len()
    }

    /// Select an option. For single-select, records the answer and advances.
    /// For multi-select, toggles the option's selected state.
    /// Returns `true` if the entire question set is now complete.
    pub fn select_option(&mut self, option_index: usize) -> bool {
        let Some(q) = self.questions.get_mut(self.current_index) else {
            return true;
        };
        if option_index >= q.options.len() {
            return self.is_complete();
        }
        if q.multiple {
            q.selected[option_index] = !q.selected[option_index];
            false
        } else {
            let label = q.options[option_index].0.clone();
            self.answers.push(vec![label]);
            self.current_index += 1;
            self.is_complete()
        }
    }

    /// Submit current multi-select answers and advance.
    /// Returns `true` if the entire question set is now complete.
    pub fn submit_current(&mut self) -> bool {
        let Some(q) = self.questions.get(self.current_index) else {
            return true;
        };
        let selected: Vec<String> = q
            .selected
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| {
                if s {
                    Some(q.options[i].0.clone())
                } else {
                    None
                }
            })
            .collect();
        let answer = if selected.is_empty() {
            vec![q.options.first().map(|o| o.0.clone()).unwrap_or_default()]
        } else {
            selected
        };
        self.answers.push(answer);
        self.current_index += 1;
        self.is_complete()
    }

    /// Access questions (for recording answers before consuming).
    pub fn questions(&self) -> &[PendingQuestionItem] {
        &self.questions
    }

    /// Access accumulated answers (for recording before consuming).
    pub fn answers(&self) -> &[Vec<String>] {
        &self.answers
    }

    /// Consume and return the collected answers.
    pub fn into_answers(self) -> Vec<Vec<String>> {
        self.answers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit::StopReason;

    #[test]
    fn pretty_print_json_formats_valid_json() {
        let input = r#"{"path":"foo.rs","content":"hello"}"#;
        let result = pretty_print_json(input);
        assert!(result.contains('\n')); // indented
        assert!(result.contains("foo.rs"));
    }

    #[test]
    fn pretty_print_json_passes_through_invalid() {
        assert_eq!(pretty_print_json("not json"), "not json");
        assert_eq!(pretty_print_json(""), "");
    }

    #[test]
    fn tool_call_input_is_pretty_printed() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::ToolCallStarted {
            agent: "a".into(),
            tool_name: "write_file".into(),
            tool_call_id: "c1".into(),
            input: r#"{"path":"test.rs","content":"fn main() {}"}"#.into(),
        });
        let tool_msg = state
            .messages()
            .iter()
            .find(|m| m.kind == MessageKind::ToolCall)
            .unwrap();
        let input = tool_msg.tool_input.as_ref().unwrap();
        // Should be pretty-printed (multi-line)
        assert!(input.contains('\n'));
        assert!(input.contains("test.rs"));
    }

    fn make_usage(input: u32, output: u32) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            reasoning_tokens: 0,
        }
    }

    #[test]
    fn run_started_creates_agent_thinking() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "researcher".into(),
            task: "find info".into(),
        });
        let agent = state.agents().get("researcher").unwrap();
        assert_eq!(agent.status, AgentStatus::Thinking);
        assert_eq!(agent.name, "researcher");
        assert_eq!(state.messages().len(), 1);
        assert_eq!(state.messages()[0].kind, MessageKind::System);
        assert!(state.messages()[0].content.contains("researcher"));
        // First agent: short message (no task text, since user message already shows it)
        assert!(!state.messages()[0].content.contains("find info"));
    }

    #[test]
    fn sub_agent_run_started_includes_task() {
        let mut state = EventProcessorState::new();
        // First agent (orchestrator)
        state.process_event(&AgentEvent::RunStarted {
            agent: "orchestrator".into(),
            task: "plan".into(),
        });
        // Sub-agent — should include the delegated task
        state.process_event(&AgentEvent::RunStarted {
            agent: "researcher".into(),
            task: "find info".into(),
        });
        let last = state.messages().last().unwrap();
        assert!(last.content.contains("researcher"));
        assert!(last.content.contains("find info"));
    }

    #[test]
    fn turn_started_updates_counters() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::TurnStarted {
            agent: "a".into(),
            turn: 3,
            max_turns: 10,
        });
        let agent = state.agents().get("a").unwrap();
        assert_eq!(agent.current_turn, 3);
        assert_eq!(agent.max_turns, 10);
    }

    #[test]
    fn llm_response_accumulates_tokens() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: make_usage(100, 50),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: String::new(),
            latency_ms: 0,
            model: None,
            time_to_first_token_ms: 0,
        });
        state.process_event(&AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 2,
            usage: make_usage(200, 75),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: String::new(),
            latency_ms: 0,
            model: None,
            time_to_first_token_ms: 0,
        });
        let agent = state.agents().get("a").unwrap();
        assert_eq!(agent.tokens_in, 300);
        assert_eq!(agent.tokens_out, 125);
        assert_eq!(state.stats().total_input, 300);
        assert_eq!(state.stats().total_output, 125);
    }

    #[test]
    fn llm_response_with_text_creates_message() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: make_usage(10, 5),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: "Hello world".into(),
            latency_ms: 42,
            model: None,
            time_to_first_token_ms: 0,
        });
        // 1 system msg from RunStarted + 1 text msg
        assert_eq!(state.messages().len(), 2);
        assert_eq!(state.messages()[1].kind, MessageKind::Text);
        assert_eq!(state.messages()[1].content, "Hello world");
    }

    #[test]
    fn llm_response_empty_text_no_message() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        let before = state.messages().len();
        state.process_event(&AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: make_usage(10, 5),
            stop_reason: StopReason::ToolUse,
            tool_call_count: 1,
            text: String::new(),
            latency_ms: 0,
            model: None,
            time_to_first_token_ms: 0,
        });
        assert_eq!(state.messages().len(), before);
    }

    #[test]
    fn tool_call_started_sets_executing() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::ToolCallStarted {
            agent: "a".into(),
            tool_name: "web_search".into(),
            tool_call_id: "c1".into(),
            input: "{}".into(),
        });
        let agent = state.agents().get("a").unwrap();
        assert_eq!(agent.status, AgentStatus::Executing);
        assert_eq!(agent.current_tool.as_deref(), Some("web_search"));
        assert_eq!(state.stats().tool_calls, 1);
    }

    #[test]
    fn tool_call_completed_reverts_thinking() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::ToolCallStarted {
            agent: "a".into(),
            tool_name: "bash".into(),
            tool_call_id: "c1".into(),
            input: "{}".into(),
        });
        state.process_event(&AgentEvent::ToolCallCompleted {
            agent: "a".into(),
            tool_name: "bash".into(),
            tool_call_id: "c1".into(),
            is_error: false,
            duration_ms: 100,
            output: "ok".into(),
        });
        let agent = state.agents().get("a").unwrap();
        assert_eq!(agent.status, AgentStatus::Thinking);
        assert!(agent.current_tool.is_none());
    }

    #[test]
    fn tool_call_completed_updates_message() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::ToolCallStarted {
            agent: "a".into(),
            tool_name: "read_file".into(),
            tool_call_id: "c1".into(),
            input: r#"{"path":"f.rs"}"#.into(),
        });
        state.process_event(&AgentEvent::ToolCallCompleted {
            agent: "a".into(),
            tool_name: "read_file".into(),
            tool_call_id: "c1".into(),
            is_error: true,
            duration_ms: 50,
            output: "file not found".into(),
        });
        // Find the tool call message
        let tool_msg = state
            .messages()
            .iter()
            .find(|m| m.kind == MessageKind::ToolCall)
            .unwrap();
        assert_eq!(tool_msg.tool_output.as_deref(), Some("file not found"));
        assert!(tool_msg.tool_is_error);
        assert_eq!(tool_msg.tool_duration_ms, 50);
    }

    #[test]
    fn approval_requested_sets_pending() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::ApprovalRequested {
            agent: "a".into(),
            turn: 1,
            tool_names: vec!["bash".into(), "write_file".into()],
        });
        let approval = state.pending_approval().unwrap();
        assert_eq!(approval.agent, "a");
        assert_eq!(approval.tool_names, vec!["bash", "write_file"]);
    }

    #[test]
    fn approval_decision_clears_pending() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::ApprovalRequested {
            agent: "a".into(),
            turn: 1,
            tool_names: vec!["bash".into()],
        });
        assert!(state.pending_approval().is_some());
        state.process_event(&AgentEvent::ApprovalDecision {
            agent: "a".into(),
            turn: 1,
            approved: true,
        });
        assert!(state.pending_approval().is_none());
    }

    #[test]
    fn sub_agents_dispatched_creates_entries() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::SubAgentsDispatched {
            agent: "orchestrator".into(),
            agents: vec!["researcher".into(), "coder".into()],
        });
        assert!(state.agents().contains_key("researcher"));
        assert!(state.agents().contains_key("coder"));
        assert_eq!(
            state.agents().get("researcher").unwrap().status,
            AgentStatus::Idle
        );
    }

    #[test]
    fn sub_agent_completed_sets_status() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::SubAgentsDispatched {
            agent: "orchestrator".into(),
            agents: vec!["a".into(), "b".into()],
        });
        state.process_event(&AgentEvent::SubAgentCompleted {
            agent: "a".into(),
            success: true,
            usage: make_usage(100, 50),
        });
        state.process_event(&AgentEvent::SubAgentCompleted {
            agent: "b".into(),
            success: false,
            usage: make_usage(80, 40),
        });
        assert_eq!(
            state.agents().get("a").unwrap().status,
            AgentStatus::Completed
        );
        assert_eq!(state.agents().get("b").unwrap().status, AgentStatus::Failed);
    }

    #[test]
    fn sub_agent_completed_does_not_double_count_tokens() {
        // Sub-agent tokens are already accumulated via forwarded LlmResponse events.
        // SubAgentCompleted should only set status, not accumulate again.
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::SubAgentsDispatched {
            agent: "orchestrator".into(),
            agents: vec!["a".into()],
        });
        // Simulate sub-agent LlmResponse (forwarded via on_event)
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "sub-task".into(),
        });
        state.process_event(&AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: make_usage(100, 50),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: String::new(),
            latency_ms: 0,
            model: None,
            time_to_first_token_ms: 0,
        });
        assert_eq!(state.stats().total_input, 100);
        assert_eq!(state.stats().total_output, 50);
        let agent = state.agents().get("a").unwrap();
        assert_eq!(agent.tokens_in, 100);
        assert_eq!(agent.tokens_out, 50);

        // SubAgentCompleted should NOT add tokens again
        state.process_event(&AgentEvent::SubAgentCompleted {
            agent: "a".into(),
            success: true,
            usage: make_usage(100, 50),
        });
        // Global stats unchanged (no double-counting)
        assert_eq!(state.stats().total_input, 100);
        assert_eq!(state.stats().total_output, 50);
    }

    #[test]
    fn run_completed_sets_completed() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        // Accumulate tokens via LlmResponse (as happens in real flow)
        state.process_event(&AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: make_usage(500, 200),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: String::new(),
            latency_ms: 0,
            model: None,
            time_to_first_token_ms: 0,
        });
        state.process_event(&AgentEvent::RunCompleted {
            agent: "a".into(),
            total_usage: make_usage(500, 200),
            tool_calls_made: 3,
        });
        let agent = state.agents().get("a").unwrap();
        assert_eq!(agent.status, AgentStatus::Completed);
        // Stats accumulated from LlmResponse (RunCompleted only sets status)
        assert_eq!(state.stats().total_input, 500);
        assert_eq!(state.stats().total_output, 200);
    }

    #[test]
    fn run_failed_sets_failed() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::RunFailed {
            agent: "a".into(),
            error: "timeout".into(),
            partial_usage: make_usage(50, 25),
        });
        let agent = state.agents().get("a").unwrap();
        assert_eq!(agent.status, AgentStatus::Failed);
        let last = state.messages().last().unwrap();
        assert_eq!(last.kind, MessageKind::System);
        assert!(last.content.contains("timeout"));
    }

    #[test]
    fn guardrail_denied_creates_system_message() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::GuardrailDenied {
            agent: "a".into(),
            hook: "pre_tool".into(),
            reason: "unsafe command".into(),
            tool_name: Some("bash".into()),
        });
        let last = state.messages().last().unwrap();
        assert_eq!(last.kind, MessageKind::System);
        assert!(last.content.contains("bash"));
        assert!(last.content.contains("unsafe command"));
    }

    #[test]
    fn context_summarized_creates_system_message() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::ContextSummarized {
            agent: "a".into(),
            turn: 5,
            usage: make_usage(10, 5),
        });
        let last = state.messages().last().unwrap();
        assert_eq!(last.kind, MessageKind::System);
        assert!(last.content.contains("turn 5"));
    }

    #[test]
    fn messages_have_sequential_ids() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t1".into(),
        });
        state.process_event(&AgentEvent::RunStarted {
            agent: "b".into(),
            task: "t2".into(),
        });
        state.process_event(&AgentEvent::GuardrailDenied {
            agent: "a".into(),
            hook: "post_llm".into(),
            reason: "bad".into(),
            tool_name: None,
        });
        let ids: Vec<i32> = state.messages().iter().map(|m| m.id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn multiple_agents_tracked_independently() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::RunStarted {
            agent: "b".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: make_usage(100, 50),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: String::new(),
            latency_ms: 0,
            model: None,
            time_to_first_token_ms: 0,
        });
        assert_eq!(state.agents().get("a").unwrap().tokens_in, 100);
        assert_eq!(state.agents().get("b").unwrap().tokens_in, 0);
    }

    #[test]
    fn unknown_agent_tool_call_handled() {
        let mut state = EventProcessorState::new();
        // Should not panic even though "unknown" was never registered
        state.process_event(&AgentEvent::ToolCallStarted {
            agent: "unknown".into(),
            tool_name: "bash".into(),
            tool_call_id: "c1".into(),
            input: "{}".into(),
        });
        // Agent not in map — tool call message still created
        assert!(state.agents().get("unknown").is_none());
        assert_eq!(state.stats().tool_calls, 1);
    }

    #[test]
    fn state_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EventProcessorState>();
    }

    #[test]
    fn guardrail_denied_without_tool_name() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::GuardrailDenied {
            agent: "a".into(),
            hook: "post_llm".into(),
            reason: "content policy".into(),
            tool_name: None,
        });
        let last = state.messages().last().unwrap();
        assert!(last.content.contains("post_llm"));
        assert!(last.content.contains("content policy"));
        assert!(!last.content.contains("tool '"));
    }

    #[test]
    fn toggle_expanded_adds_and_removes() {
        let mut state = EventProcessorState::new();
        assert!(state.toggle_expanded(42)); // first toggle → true (expanded)
        assert!(state.is_expanded(42));
        assert!(!state.toggle_expanded(42)); // second toggle → false (collapsed)
        assert!(!state.is_expanded(42));
    }

    #[test]
    fn is_expanded_default_false() {
        let state = EventProcessorState::new();
        assert!(!state.is_expanded(0));
        assert!(!state.is_expanded(999));
    }

    #[test]
    fn expansion_survives_event() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::ToolCallStarted {
            agent: "a".into(),
            tool_name: "bash".into(),
            tool_call_id: "c1".into(),
            input: "{}".into(),
        });
        // Expand the tool call message (id=1, after system msg id=0)
        let tool_msg_id = state
            .messages()
            .iter()
            .find(|m| m.kind == MessageKind::ToolCall)
            .unwrap()
            .id;
        state.toggle_expanded(tool_msg_id);
        assert!(state.is_expanded(tool_msg_id));

        // Process another event
        state.process_event(&AgentEvent::ToolCallCompleted {
            agent: "a".into(),
            tool_name: "bash".into(),
            tool_call_id: "c1".into(),
            is_error: false,
            duration_ms: 100,
            output: "ok".into(),
        });
        // Expansion survives
        assert!(state.is_expanded(tool_msg_id));
    }

    #[test]
    fn streaming_text_creates_message() {
        let mut state = EventProcessorState::new();
        state.update_streaming_text("agent", "Hello");
        assert_eq!(state.messages().len(), 1);
        assert_eq!(state.messages()[0].content, "Hello");
        assert_eq!(state.messages()[0].kind, MessageKind::Text);
    }

    #[test]
    fn streaming_accumulates_deltas() {
        let mut state = EventProcessorState::new();
        state.update_streaming_text("agent", "Hello");
        state.update_streaming_text("agent", " world");
        assert_eq!(state.messages().len(), 1);
        assert_eq!(state.messages()[0].content, "Hello world");
    }

    #[test]
    fn llm_response_keeps_streaming_message() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.update_streaming_text("a", "full streaming content");
        // system msg + streaming msg = 2
        assert_eq!(state.messages().len(), 2);

        state.process_event(&AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: make_usage(10, 5),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            // Event text is truncated — should NOT replace full streaming content
            text: "full s...".into(),
            latency_ms: 42,
            model: Some("claude-4".into()),
            time_to_first_token_ms: 0,
        });
        // Streaming message kept, no new message created → still 2
        assert_eq!(state.messages().len(), 2);
        let last = state.messages().last().unwrap();
        // Full streaming content preserved (not the truncated event text)
        assert_eq!(last.content, "full streaming content");
    }

    #[test]
    fn llm_response_creates_message_without_streaming() {
        // When no streaming occurred (e.g., non-streaming provider), LlmResponse
        // should create a message from the event text.
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: make_usage(10, 5),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: "response without streaming".into(),
            latency_ms: 42,
            model: None,
            time_to_first_token_ms: 0,
        });
        // system msg + text msg = 2
        assert_eq!(state.messages().len(), 2);
        assert_eq!(state.messages()[1].content, "response without streaming");
    }

    #[test]
    fn set_estimated_cost_updates_stats() {
        let mut state = EventProcessorState::new();
        assert_eq!(state.stats().estimated_cost, 0.0);
        state.set_estimated_cost(0.0042);
        assert!((state.stats().estimated_cost - 0.0042).abs() < f64::EPSILON);
    }

    #[test]
    fn llm_response_captures_latency_model() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: make_usage(10, 5),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: String::new(),
            latency_ms: 1234,
            model: Some("claude-sonnet-4".into()),
            time_to_first_token_ms: 0,
        });
        assert_eq!(state.stats().last_latency_ms, 1234);
        assert_eq!(state.stats().model.as_deref(), Some("claude-sonnet-4"));
    }

    #[test]
    fn set_blackboard_entries_stored() {
        let mut state = EventProcessorState::new();
        state.set_blackboard_entries(vec![
            ("key1".into(), "val1".into()),
            ("key2".into(), "val2".into()),
        ]);
        assert_eq!(state.blackboard_entries().len(), 2);
        assert_eq!(
            state.blackboard_entries()[0],
            ("key1".into(), "val1".into())
        );
    }

    #[test]
    fn blackboard_entries_empty_by_default() {
        let state = EventProcessorState::new();
        assert!(state.blackboard_entries().is_empty());
    }

    #[test]
    fn current_agent_tracks_run_started() {
        let mut state = EventProcessorState::new();
        assert_eq!(state.current_agent(), "heartbit"); // default
        state.process_event(&AgentEvent::RunStarted {
            agent: "researcher".into(),
            task: "t".into(),
        });
        assert_eq!(state.current_agent(), "researcher");
    }

    #[test]
    fn reset_clears_all_state() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: make_usage(100, 50),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: "hello".into(),
            latency_ms: 42,
            model: Some("claude-4".into()),
            time_to_first_token_ms: 0,
        });
        state.toggle_expanded(0);
        state.set_blackboard_entries(vec![("k".into(), "v".into())]);
        state.set_estimated_cost(0.01);

        state.reset();

        assert!(state.agents().is_empty());
        assert!(state.messages().is_empty());
        assert!(state.pending_approval().is_none());
        assert_eq!(state.stats().total_input, 0);
        assert_eq!(state.stats().estimated_cost, 0.0);
        assert!(!state.is_expanded(0));
        assert!(state.blackboard_entries().is_empty());
        assert_eq!(state.current_agent(), "heartbit");
        assert!(state.agent_filter().is_none());
    }

    #[test]
    fn toggle_agent_filter_sets_and_clears() {
        let mut state = EventProcessorState::new();
        assert!(state.agent_filter().is_none());

        state.toggle_agent_filter("researcher");
        assert_eq!(state.agent_filter(), Some("researcher"));

        // Same agent again — clears the filter
        state.toggle_agent_filter("researcher");
        assert!(state.agent_filter().is_none());

        // Different agent — sets new filter
        state.toggle_agent_filter("coder");
        assert_eq!(state.agent_filter(), Some("coder"));
        state.toggle_agent_filter("researcher");
        assert_eq!(state.agent_filter(), Some("researcher"));
    }

    #[test]
    fn reset_clears_agent_filter() {
        let mut state = EventProcessorState::new();
        state.toggle_agent_filter("researcher");
        state.push_user_message("task");
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "task".into(),
        });
        state.reset();
        assert!(state.agent_filter().is_none());
    }

    #[test]
    fn push_user_message_creates_user_kind() {
        let mut state = EventProcessorState::new();
        state.push_user_message("hello from user");
        assert_eq!(state.messages().len(), 1);
        assert_eq!(state.messages()[0].kind, MessageKind::User);
        assert_eq!(state.messages()[0].content, "hello from user");
        assert!(state.messages()[0].agent.is_empty());
    }

    #[test]
    fn set_current_agent_idle_updates_status() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        assert_eq!(
            state.agents().get("a").unwrap().status,
            AgentStatus::Thinking
        );
        state.set_current_agent_idle();
        assert_eq!(state.agents().get("a").unwrap().status, AgentStatus::Idle);
    }

    #[test]
    fn turn_started_updates_current_agent() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "orchestrator".into(),
            task: "plan".into(),
        });
        state.process_event(&AgentEvent::RunStarted {
            agent: "researcher".into(),
            task: "search".into(),
        });
        assert_eq!(state.current_agent(), "researcher");
        // Orchestrator resumes — TurnStarted should switch current_agent back
        state.process_event(&AgentEvent::TurnStarted {
            agent: "orchestrator".into(),
            turn: 2,
            max_turns: 10,
        });
        assert_eq!(state.current_agent(), "orchestrator");
    }

    #[test]
    fn running_cost_computed_from_tokens_and_model() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: make_usage(1000, 500),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: "hi".into(),
            latency_ms: 100,
            model: Some("anthropic/claude-sonnet-4".into()),
            time_to_first_token_ms: 0,
        });
        // Cost should be non-zero after LlmResponse with known model
        assert!(state.stats().estimated_cost > 0.0);
    }

    #[test]
    fn cancel_active_agents_sets_cancelled() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::RunStarted {
            agent: "b".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::ToolCallStarted {
            agent: "b".into(),
            tool_name: "bash".into(),
            tool_call_id: "c1".into(),
            input: "{}".into(),
        });
        // a = Thinking, b = Executing
        assert_eq!(
            state.agents().get("a").unwrap().status,
            AgentStatus::Thinking
        );
        assert_eq!(
            state.agents().get("b").unwrap().status,
            AgentStatus::Executing
        );

        state.cancel_active_agents();

        assert_eq!(
            state.agents().get("a").unwrap().status,
            AgentStatus::Cancelled
        );
        assert_eq!(
            state.agents().get("b").unwrap().status,
            AgentStatus::Cancelled
        );
        // Tool should be cleared
        assert!(state.agents().get("b").unwrap().current_tool.is_none());
    }

    #[test]
    fn cancel_finalizes_streaming_message() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.update_streaming_text("a", "partial response");
        state.cancel_active_agents();
        // Streaming message should be finalized with [cancelled] marker
        let text_msg = state
            .messages()
            .iter()
            .find(|m| m.kind == MessageKind::Text)
            .unwrap();
        assert!(text_msg.content.ends_with("[cancelled]"));
        // Streaming state should be cleared
        assert!(state.streaming_message_id.is_none());
        assert!(state.streaming_buffer.is_empty());
    }

    #[test]
    fn cancel_active_agents_preserves_terminal() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        state.process_event(&AgentEvent::RunCompleted {
            agent: "a".into(),
            total_usage: make_usage(100, 50),
            tool_calls_made: 0,
        });
        assert_eq!(
            state.agents().get("a").unwrap().status,
            AgentStatus::Completed
        );

        state.cancel_active_agents();

        // Completed agents should NOT change to cancelled
        assert_eq!(
            state.agents().get("a").unwrap().status,
            AgentStatus::Completed
        );
    }

    #[test]
    fn clear_pending_approval_clears_state() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::ApprovalRequested {
            agent: "a".into(),
            turn: 1,
            tool_names: vec!["bash".into()],
        });
        assert!(state.pending_approval().is_some());
        state.clear_pending_approval();
        assert!(state.pending_approval().is_none());
    }

    #[test]
    fn elapsed_seconds_zero_before_run() {
        let state = EventProcessorState::new();
        assert_eq!(state.elapsed_seconds(), 0);
    }

    #[test]
    fn elapsed_seconds_nonzero_after_run_started() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        // run_start was just set, elapsed should be 0 (sub-second)
        assert!(state.elapsed_seconds() < 2);
        assert!(state.stats.run_start.is_some());
    }

    #[test]
    fn freeze_elapsed_stops_timer() {
        let mut state = EventProcessorState::new();
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        });
        assert!(state.stats.run_start.is_some());

        state.freeze_elapsed();

        // Timer stopped but frozen value preserved
        assert!(state.stats.run_start.is_none());
        // Frozen value should be 0 (sub-second since start)
        assert_eq!(state.elapsed_seconds(), 0);
    }

    #[test]
    fn push_system_message_creates_unattributed_message() {
        let mut state = EventProcessorState::new();
        state.push_system_message("Task cancelled by user");
        assert_eq!(state.messages().len(), 1);
        assert_eq!(state.messages()[0].kind, MessageKind::System);
        assert_eq!(state.messages()[0].content, "Task cancelled by user");
        assert!(state.messages()[0].agent.is_empty());
    }

    #[test]
    fn push_question_answer_creates_user_message() {
        let mut state = EventProcessorState::new();
        state.push_question_answer("Color", &["Blue".into()]);
        assert_eq!(state.messages().len(), 1);
        assert_eq!(state.messages()[0].kind, MessageKind::User);
        assert_eq!(state.messages()[0].content, "Color: Blue");
    }

    // --- PendingQuestion tests ---

    fn make_single_select_item(header: &str, options: &[&str]) -> PendingQuestionItem {
        PendingQuestionItem {
            header: header.into(),
            question: format!("Pick a {header}"),
            options: options
                .iter()
                .map(|&l| (l.into(), format!("{l} description")))
                .collect(),
            multiple: false,
            selected: vec![false; options.len()],
        }
    }

    fn make_multi_select_item(header: &str, options: &[&str]) -> PendingQuestionItem {
        PendingQuestionItem {
            header: header.into(),
            question: format!("Select {header}"),
            options: options
                .iter()
                .map(|&l| (l.into(), format!("{l} description")))
                .collect(),
            multiple: true,
            selected: vec![false; options.len()],
        }
    }

    #[test]
    fn pending_question_single_select() {
        let items = vec![make_single_select_item("Color", &["Red", "Blue"])];
        let mut pq = PendingQuestion::new(items);
        assert!(!pq.is_complete());
        assert!(pq.current().is_some());

        let complete = pq.select_option(1);
        assert!(complete);
        assert_eq!(pq.into_answers(), vec![vec!["Blue".to_string()]]);
    }

    #[test]
    fn pending_question_multi_select() {
        let items = vec![make_multi_select_item("Features", &["A", "B", "C"])];
        let mut pq = PendingQuestion::new(items);

        // Toggle A and C
        assert!(!pq.select_option(0));
        assert!(!pq.select_option(2));

        let complete = pq.submit_current();
        assert!(complete);
        assert_eq!(
            pq.into_answers(),
            vec![vec!["A".to_string(), "C".to_string()]]
        );
    }

    #[test]
    fn pending_question_multiple_questions() {
        let items = vec![
            make_single_select_item("Q1", &["X", "Y"]),
            make_single_select_item("Q2", &["A", "B"]),
        ];
        let mut pq = PendingQuestion::new(items);
        assert_eq!(pq.progress_text(), "1/2");

        let complete = pq.select_option(0); // answer Q1 with "X"
        assert!(!complete);
        assert_eq!(pq.progress_text(), "2/2");

        let complete = pq.select_option(1); // answer Q2 with "B"
        assert!(complete);
        assert_eq!(
            pq.into_answers(),
            vec![vec!["X".to_string()], vec!["B".to_string()]]
        );
    }

    #[test]
    fn pending_question_multi_select_default_first() {
        let items = vec![make_multi_select_item("H", &["Default", "Other"])];
        let mut pq = PendingQuestion::new(items);
        // Submit without selecting anything → defaults to first
        pq.submit_current();
        assert_eq!(pq.into_answers(), vec![vec!["Default".to_string()]]);
    }

    #[test]
    fn pending_question_progress_text_single() {
        let items = vec![make_single_select_item("H", &["A", "B"])];
        let pq = PendingQuestion::new(items);
        assert_eq!(pq.progress_text(), ""); // single question: no progress
    }

    #[test]
    fn pending_question_out_of_bounds_option() {
        let items = vec![make_single_select_item("H", &["A", "B"])];
        let mut pq = PendingQuestion::new(items);
        let complete = pq.select_option(99);
        assert!(!complete); // out of bounds ignored
        assert!(!pq.is_complete());
    }

    #[test]
    fn pending_question_multi_toggle_deselect() {
        let items = vec![make_multi_select_item("H", &["A", "B"])];
        let mut pq = PendingQuestion::new(items);
        pq.select_option(0); // select A
        pq.select_option(0); // deselect A
        pq.select_option(1); // select B
        pq.submit_current();
        assert_eq!(pq.into_answers(), vec![vec!["B".to_string()]]);
    }

    // --- History tests ---

    #[test]
    fn reset_archives_current_run() {
        let mut state = EventProcessorState::new();
        state.push_user_message("do something");
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "do something".into(),
        });
        state.process_event(&AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: make_usage(100, 50),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: "done".into(),
            latency_ms: 0,
            model: None,
            time_to_first_token_ms: 0,
        });
        state.process_event(&AgentEvent::RunCompleted {
            agent: "a".into(),
            total_usage: make_usage(100, 50),
            tool_calls_made: 0,
        });
        state.freeze_elapsed();

        // Reset archives the run
        state.reset();
        assert_eq!(state.history().len(), 1);
        assert_eq!(state.history()[0].task, "do something");
        assert_eq!(state.history()[0].status, "completed");
        assert!(state.history()[0].message_count > 1);
    }

    #[test]
    fn reset_does_not_archive_empty_run() {
        let mut state = EventProcessorState::new();
        // Only a user message — no agent events
        state.push_user_message("hello");
        state.reset();
        // Single message (user msg) is not enough to archive
        assert!(state.history().is_empty());
    }

    #[test]
    fn reset_does_not_archive_no_messages() {
        let mut state = EventProcessorState::new();
        state.reset();
        assert!(state.history().is_empty());
    }

    #[test]
    fn view_history_switches_visible_agents() {
        let mut state = EventProcessorState::new();
        state.push_user_message("task1");
        state.process_event(&AgentEvent::RunStarted {
            agent: "researcher".into(),
            task: "task1".into(),
        });
        state.process_event(&AgentEvent::RunCompleted {
            agent: "researcher".into(),
            total_usage: make_usage(100, 50),
            tool_calls_made: 0,
        });
        state.reset();

        // New run with a different agent
        state.push_user_message("task2");
        state.process_event(&AgentEvent::RunStarted {
            agent: "coder".into(),
            task: "task2".into(),
        });

        // Live view: only "coder"
        assert!(state.visible_agents().contains_key("coder"));
        assert!(!state.visible_agents().contains_key("researcher"));

        // History view: only "researcher"
        state.view_history(Some(0));
        assert!(state.visible_agents().contains_key("researcher"));
        assert!(!state.visible_agents().contains_key("coder"));

        // Back to live
        state.view_history(None);
        assert!(state.visible_agents().contains_key("coder"));
    }

    #[test]
    fn view_history_switches_visible_messages() {
        let mut state = EventProcessorState::new();
        state.push_user_message("first task");
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "first task".into(),
        });
        state.process_event(&AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: make_usage(100, 50),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: "first response".into(),
            latency_ms: 0,
            model: None,
            time_to_first_token_ms: 0,
        });
        state.reset();

        // Now add new messages for current run
        state.push_user_message("second task");
        assert_eq!(state.visible_messages().len(), 1); // live view: just user msg

        // Switch to history view
        state.view_history(Some(0));
        assert!(state.visible_messages().len() > 1); // archived messages
        assert_eq!(state.viewing_history(), Some(0));

        // Switch back to live
        state.view_history(None);
        assert_eq!(state.visible_messages().len(), 1);
    }

    #[test]
    fn view_history_clears_expansion_state() {
        let mut state = EventProcessorState::new();
        // Create a run with a tool call, expand it
        state.push_user_message("task");
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "task".into(),
        });
        state.process_event(&AgentEvent::ToolCallStarted {
            agent: "a".into(),
            tool_name: "bash".into(),
            tool_call_id: "c1".into(),
            input: "{}".into(),
        });
        let tool_msg_id = state
            .messages()
            .iter()
            .find(|m| m.kind == MessageKind::ToolCall)
            .unwrap()
            .id;
        state.toggle_expanded(tool_msg_id);
        assert!(state.is_expanded(tool_msg_id));
        state.reset();

        // Start a second run — message IDs restart at 0
        state.push_user_message("task2");
        state.process_event(&AgentEvent::RunStarted {
            agent: "a".into(),
            task: "task2".into(),
        });

        // Expansion state was cleared by reset()
        assert!(!state.is_expanded(tool_msg_id));

        // Expand something in live view, then switch to history — should clear
        state.toggle_expanded(0);
        assert!(state.is_expanded(0));
        state.view_history(Some(0));
        assert!(!state.is_expanded(0));

        // Expand something in history view, then switch back — should clear
        state.toggle_expanded(1);
        assert!(state.is_expanded(1));
        state.view_history(None);
        assert!(!state.is_expanded(1));
    }

    #[test]
    fn multiple_runs_accumulate_history() {
        let mut state = EventProcessorState::new();
        for i in 0..3 {
            state.push_user_message(&format!("task {i}"));
            state.process_event(&AgentEvent::RunStarted {
                agent: "a".into(),
                task: format!("task {i}"),
            });
            state.reset();
        }
        assert_eq!(state.history().len(), 3);
        assert_eq!(state.history()[0].task, "task 0");
        assert_eq!(state.history()[2].task, "task 2");
    }
}
