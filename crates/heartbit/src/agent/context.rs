use crate::llm::types::{
    CompletionRequest, ContentBlock, Message, Role, ToolDefinition, ToolResult,
};

use super::token_estimator::{estimate_message_tokens, estimate_tokens};

/// Strategy for managing the context window.
#[derive(Debug, Clone)]
pub enum ContextStrategy {
    /// No trimming — all messages are sent (current default behavior).
    Unlimited,
    /// Keep first message + as many recent messages as fit in `max_tokens`.
    SlidingWindow { max_tokens: u32 },
}

/// Conversation context for an agent run.
pub(crate) struct AgentContext {
    system: String,
    messages: Vec<Message>,
    tools: Vec<ToolDefinition>,
    max_turns: usize,
    max_tokens: u32,
    current_turn: usize,
    context_strategy: ContextStrategy,
}

impl AgentContext {
    pub(crate) fn new(
        system: impl Into<String>,
        task: impl Into<String>,
        tools: Vec<ToolDefinition>,
    ) -> Self {
        Self {
            system: system.into(),
            messages: vec![Message::user(task)],
            tools,
            max_turns: 10,
            max_tokens: 4096,
            current_turn: 0,
            context_strategy: ContextStrategy::Unlimited,
        }
    }

    pub(crate) fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    pub(crate) fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub(crate) fn with_context_strategy(mut self, strategy: ContextStrategy) -> Self {
        self.context_strategy = strategy;
        self
    }

    pub(crate) fn current_turn(&self) -> usize {
        self.current_turn
    }

    pub(crate) fn max_turns(&self) -> usize {
        self.max_turns
    }

    pub(crate) fn increment_turn(&mut self) {
        self.current_turn += 1;
    }

    pub(crate) fn add_assistant_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub(crate) fn add_tool_results(&mut self, results: Vec<ToolResult>) {
        self.messages.push(Message::tool_results(results));
    }

    /// Get the text from the last assistant message (avoids re-cloning the response).
    pub(crate) fn last_assistant_text(&self) -> Option<String> {
        self.messages.iter().rev().find_map(|m| {
            if m.role == Role::Assistant {
                let text: String = m
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                Some(text)
            } else {
                None
            }
        })
    }

    /// Estimate total tokens across all messages.
    pub(crate) fn total_tokens(&self) -> u32 {
        self.messages
            .iter()
            .map(estimate_message_tokens)
            .sum::<u32>()
            + estimate_tokens(&self.system)
    }

    /// Check whether the context exceeds a token threshold and needs compaction.
    pub(crate) fn needs_compaction(&self, max_tokens: u32) -> bool {
        self.total_tokens() > max_tokens
    }

    /// Replace old messages with a summary, keeping the initial task context
    /// and the last `keep_last_n` messages.
    ///
    /// The summary is merged into the first user message to maintain the required
    /// alternating user/assistant role sequence (Anthropic API constraint).
    ///
    /// If there aren't enough messages to compact (first + keep_last_n >= total),
    /// this is a no-op.
    pub(crate) fn inject_summary(&mut self, summary: String, keep_last_n: usize) {
        // Extract the original task text from the first message
        let Some(first) = self.messages.first() else {
            return;
        };
        let original_task: String = first
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        inject_summary_into_messages(&mut self.messages, &original_task, &summary, keep_last_n);
    }

    /// Render all messages as a plain text transcript for summarization.
    pub(crate) fn conversation_text(&self) -> String {
        messages_to_text(&self.messages)
    }

    pub(crate) fn to_request(&self) -> CompletionRequest {
        let messages = match &self.context_strategy {
            ContextStrategy::Unlimited => self.messages.clone(),
            ContextStrategy::SlidingWindow { max_tokens } => {
                apply_sliding_window(&self.messages, *max_tokens)
            }
        };

        CompletionRequest {
            system: self.system.clone(),
            messages,
            tools: self.tools.clone(),
            max_tokens: self.max_tokens,
        }
    }
}

/// Inject a summary into a message list, replacing middle messages.
///
/// Keeps the original task from `messages[0]` and merges it with the summary
/// into a single User message. Then appends the last `keep_last_n` messages.
/// Adjusts the tail start to ensure User/Assistant alternation is preserved.
///
/// Shared between standalone (`AgentContext`) and durable (`AgentWorkflow`) paths.
pub(crate) fn inject_summary_into_messages(
    messages: &mut Vec<Message>,
    original_task: &str,
    summary: &str,
    keep_last_n: usize,
) {
    if messages.is_empty() {
        return;
    }
    let total = messages.len();
    // Need at least: first(1) + something_to_summarize(1) + keep_last_n
    if total <= 1 + keep_last_n {
        return;
    }

    let combined = Message::user(format!(
        "{original_task}\n\n[Previous conversation summary]\n{summary}"
    ));

    // Determine tail start, then adjust to maintain alternating User/Assistant roles.
    // After the combined User message, the tail must start with an Assistant message.
    let mut tail_start = total.saturating_sub(keep_last_n);
    if tail_start < total && messages[tail_start].role == Role::User && tail_start > 1 {
        tail_start -= 1;
    }
    let last_messages: Vec<Message> = messages[tail_start..].to_vec();

    messages.clear();
    messages.push(combined);
    messages.extend(last_messages);
}

/// Render a message list as a plain text transcript for summarization.
///
/// Shared between standalone (`AgentContext`) and durable (`AgentWorkflow`) paths.
pub(crate) fn messages_to_text(messages: &[Message]) -> String {
    let mut parts = Vec::with_capacity(messages.len());
    for msg in messages {
        let role = match msg.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
        };
        let text: String = msg
            .content
            .iter()
            .map(|b| match b {
                ContentBlock::Text { text } => text.as_str().into(),
                ContentBlock::ToolUse { name, input, .. } => {
                    format!("[Tool call: {name}({input})]")
                }
                ContentBlock::ToolResult { content, .. } => {
                    format!("[Tool result: {content}]")
                }
            })
            .collect::<Vec<String>>()
            .join(" ");
        parts.push(format!("{role}: {text}"));
    }
    parts.join("\n")
}

/// Apply sliding window to a message list: always keep the first message (initial task),
/// then include as many recent messages as fit within `max_tokens`.
///
/// Tool use/result pairs are kept together to avoid orphaned tool references.
///
/// Shared between standalone (`AgentContext`) and durable (`AgentWorkflow`) paths.
pub(crate) fn apply_sliding_window(messages: &[Message], max_tokens: u32) -> Vec<Message> {
    if messages.len() <= 1 {
        return messages.to_vec();
    }

    let first = &messages[0];
    let first_tokens = estimate_message_tokens(first);
    if first_tokens >= max_tokens {
        return vec![first.clone()];
    }

    let mut budget = max_tokens - first_tokens;
    let tail = &messages[1..];

    // Walk backward, accumulating messages. Keep tool_use/tool_result pairs together.
    let mut included_from = tail.len();
    let mut i = tail.len();
    while i > 0 {
        i -= 1;
        let msg = &tail[i];
        let msg_tokens = estimate_message_tokens(msg);

        // Check if this message is a tool_result (User with ToolResult blocks)
        // and the previous message is the corresponding tool_use (Assistant with ToolUse).
        // If so, they must be included together.
        let is_tool_result = msg.role == Role::User
            && msg
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolResult { .. }));

        if is_tool_result && i > 0 {
            let prev = &tail[i - 1];
            let prev_tokens = estimate_message_tokens(prev);
            let pair_tokens = msg_tokens + prev_tokens;

            if pair_tokens <= budget {
                budget -= pair_tokens;
                i -= 1;
                included_from = i;
            } else {
                break;
            }
        } else if msg_tokens <= budget {
            budget -= msg_tokens;
            included_from = i;
        } else {
            break;
        }
    }

    let mut result = vec![first.clone()];
    result.extend_from_slice(&tail[included_from..]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn new_context_has_user_message() {
        let ctx = AgentContext::new("system", "do something", vec![]);
        let req = ctx.to_request();

        assert_eq!(req.system, "system");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, Role::User);
    }

    #[test]
    fn with_max_turns_overrides_default() {
        let ctx = AgentContext::new("sys", "task", vec![]).with_max_turns(5);
        assert_eq!(ctx.max_turns(), 5);
    }

    #[test]
    fn with_max_tokens_overrides_default() {
        let ctx = AgentContext::new("sys", "task", vec![]).with_max_tokens(8192);
        let req = ctx.to_request();
        assert_eq!(req.max_tokens, 8192);
    }

    #[test]
    fn default_max_tokens_is_4096() {
        let ctx = AgentContext::new("sys", "task", vec![]);
        let req = ctx.to_request();
        assert_eq!(req.max_tokens, 4096);
    }

    #[test]
    fn turn_tracking() {
        let mut ctx = AgentContext::new("sys", "task", vec![]);
        assert_eq!(ctx.current_turn(), 0);
        ctx.increment_turn();
        assert_eq!(ctx.current_turn(), 1);
    }

    #[test]
    fn add_tool_results_creates_user_message() {
        let mut ctx = AgentContext::new("sys", "task", vec![]);
        ctx.add_tool_results(vec![ToolResult::success("call-1", "result")]);

        let req = ctx.to_request();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[1].role, Role::User);
    }

    #[test]
    fn request_includes_tools() {
        let tools = vec![ToolDefinition {
            name: "search".into(),
            description: "Search".into(),
            input_schema: json!({"type": "object"}),
        }];
        let ctx = AgentContext::new("sys", "task", tools);
        let req = ctx.to_request();
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0].name, "search");
    }

    #[test]
    fn default_is_unlimited() {
        let ctx = AgentContext::new("sys", "task", vec![]);
        assert!(matches!(ctx.context_strategy, ContextStrategy::Unlimited));
    }

    #[test]
    fn unlimited_passes_all() {
        let mut ctx = AgentContext::new("sys", "task", vec![]);
        ctx.add_assistant_message(Message::assistant("response 1"));
        ctx.add_assistant_message(Message::assistant("response 2"));
        ctx.add_assistant_message(Message::assistant("response 3"));

        let req = ctx.to_request();
        assert_eq!(req.messages.len(), 4); // 1 user + 3 assistant
    }

    #[test]
    fn sliding_window_preserves_first() {
        let mut ctx = AgentContext::new("sys", "initial task", vec![])
            .with_context_strategy(ContextStrategy::SlidingWindow { max_tokens: 20 });

        ctx.add_assistant_message(Message::assistant("a".repeat(100)));
        ctx.add_assistant_message(Message::assistant("recent"));

        let req = ctx.to_request();
        // First message must always be preserved
        assert_eq!(req.messages[0].role, Role::User);
        assert!(
            req.messages[0]
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Text { text } if text == "initial task"))
        );
    }

    #[test]
    fn sliding_window_trims_old() {
        let mut ctx = AgentContext::new("sys", "task", vec![])
            .with_context_strategy(ContextStrategy::SlidingWindow { max_tokens: 50 });

        // Add many messages to exceed the window
        for i in 0..10 {
            ctx.add_assistant_message(Message::assistant(format!("response {i} with some text")));
        }

        let req = ctx.to_request();
        // Should have fewer messages than the full 11
        assert!(req.messages.len() < 11);
        // First message always preserved
        assert_eq!(req.messages[0].role, Role::User);
    }

    #[test]
    fn sliding_window_keeps_tool_pairs() {
        let mut ctx = AgentContext::new("sys", "task", vec![])
            .with_context_strategy(ContextStrategy::SlidingWindow { max_tokens: 200 });

        // Add a tool use + result pair
        ctx.add_assistant_message(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "c1".into(),
                name: "search".into(),
                input: json!({"q": "test"}),
            }],
        });
        ctx.add_tool_results(vec![ToolResult::success("c1", "found it")]);
        ctx.add_assistant_message(Message::assistant("Based on the search results..."));

        let req = ctx.to_request();
        // Check that tool_use and tool_result are both present or both absent
        let has_tool_use = req.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
        });
        let has_tool_result = req.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
        });
        assert_eq!(
            has_tool_use, has_tool_result,
            "tool_use and tool_result must be kept together"
        );
    }

    #[test]
    fn sliding_window_single_message() {
        let ctx = AgentContext::new("sys", "task", vec![])
            .with_context_strategy(ContextStrategy::SlidingWindow { max_tokens: 10 });

        let req = ctx.to_request();
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn needs_compaction_below_threshold() {
        let ctx = AgentContext::new("sys", "task", vec![]);
        assert!(!ctx.needs_compaction(10000));
    }

    #[test]
    fn needs_compaction_above_threshold() {
        let mut ctx = AgentContext::new("sys", "task", vec![]);
        for _ in 0..50 {
            ctx.add_assistant_message(Message::assistant("a".repeat(200)));
        }
        assert!(ctx.needs_compaction(100));
    }

    #[test]
    fn inject_summary_replaces_middle() {
        let mut ctx = AgentContext::new("sys", "initial task", vec![]);
        ctx.add_assistant_message(Message::assistant("msg 1"));
        ctx.add_assistant_message(Message::assistant("msg 2"));
        ctx.add_assistant_message(Message::assistant("msg 3"));
        ctx.add_assistant_message(Message::assistant("msg 4"));
        ctx.add_assistant_message(Message::assistant("msg 5"));

        ctx.inject_summary("summary of earlier conversation".into(), 2);

        // Should have: combined_first(1) + last 2 = 3 messages
        assert_eq!(ctx.messages.len(), 3);
        // First message contains both original task and summary
        let first_text: String = ctx.messages[0]
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        assert!(first_text.contains("initial task"));
        assert!(first_text.contains("summary of earlier"));
    }

    #[test]
    fn inject_summary_preserves_first_and_last() {
        let mut ctx = AgentContext::new("sys", "first task", vec![]);
        ctx.add_assistant_message(Message::assistant("old 1"));
        ctx.add_assistant_message(Message::assistant("old 2"));
        ctx.add_assistant_message(Message::assistant("recent 1"));
        ctx.add_assistant_message(Message::assistant("recent 2"));
        ctx.add_assistant_message(Message::assistant("recent 3"));

        ctx.inject_summary("compressed".into(), 3);

        // combined_first(1) + last 3 = 4
        assert_eq!(ctx.messages.len(), 4);
        // Last message should be "recent 3"
        assert!(
            ctx.messages[3]
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Text { text } if text == "recent 3"))
        );
    }

    #[test]
    fn inject_summary_noop_few_messages() {
        let mut ctx = AgentContext::new("sys", "task", vec![]);
        ctx.add_assistant_message(Message::assistant("only one"));

        ctx.inject_summary("summary".into(), 4);

        // Not enough messages to compact (total=2, need > 1 + 4)
        assert_eq!(ctx.messages.len(), 2);
    }

    #[test]
    fn inject_summary_maintains_alternating_roles() {
        // After summarization, message roles must alternate (user/assistant)
        let mut ctx = AgentContext::new("sys", "task", vec![]);
        ctx.add_assistant_message(Message::assistant("a1"));
        ctx.add_assistant_message(Message::assistant("a2"));
        ctx.add_assistant_message(Message::assistant("a3"));
        ctx.add_assistant_message(Message::assistant("a4"));

        ctx.inject_summary("summary".into(), 2);

        // First message is User (combined task+summary)
        assert_eq!(ctx.messages[0].role, Role::User);
        // The remaining messages should start with assistant
        assert_eq!(ctx.messages[1].role, Role::Assistant);
    }

    #[test]
    fn inject_summary_adjusts_tail_when_starting_with_user() {
        // Regression: if keep_last_n tail starts with a User message (tool_result),
        // the combined User + User sequence violates the alternating-role invariant.
        // inject_summary must include the preceding Assistant to maintain alternation.
        let mut ctx = AgentContext::new("sys", "task", vec![]);
        ctx.add_assistant_message(Message::assistant("a1"));
        ctx.add_tool_results(vec![ToolResult::success("c1", "result1")]);
        ctx.add_assistant_message(Message::assistant("a2"));
        ctx.add_tool_results(vec![ToolResult::success("c2", "result2")]);
        ctx.add_assistant_message(Message::assistant("a3"));
        // Messages: User, A, U(tool), A, U(tool), A
        // Total = 6, keep_last_n = 2 → tail_start = 4 → messages[4] = U(tool)
        // Without fix: combined(U) + U(tool) + A → role violation
        // With fix: combined(U) + A + U(tool) + A → correct alternation

        ctx.inject_summary("summary".into(), 2);

        // First must be User, second must be Assistant
        assert_eq!(ctx.messages[0].role, Role::User);
        assert_eq!(ctx.messages[1].role, Role::Assistant);
        // Verify alternation throughout
        for w in ctx.messages.windows(2) {
            assert_ne!(w[0].role, w[1].role, "adjacent messages have same role");
        }
    }

    #[test]
    fn total_tokens_grows_with_messages() {
        let mut ctx = AgentContext::new("sys", "task", vec![]);
        let initial = ctx.total_tokens();

        ctx.add_assistant_message(Message::assistant("a".repeat(100)));
        assert!(ctx.total_tokens() > initial);
    }

    #[test]
    fn shared_inject_summary_preserves_alternation() {
        // Test the shared function directly (used by both standalone and Restate paths)
        let mut messages = vec![
            Message::user("original task"),
            Message::assistant("a1"),
            Message::tool_results(vec![ToolResult::success("c1", "result1")]),
            Message::assistant("a2"),
            Message::tool_results(vec![ToolResult::success("c2", "result2")]),
            Message::assistant("a3"),
        ];

        inject_summary_into_messages(&mut messages, "original task", "summary of conversation", 2);

        // First must be User (combined), then alternating
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[1].role, Role::Assistant);
        for w in messages.windows(2) {
            assert_ne!(w[0].role, w[1].role, "adjacent messages have same role");
        }
        // Combined message contains both task and summary
        let first_text: String = messages[0]
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        assert!(first_text.contains("original task"));
        assert!(first_text.contains("summary of conversation"));
    }

    #[test]
    fn inject_summary_empty_messages_is_noop() {
        let mut messages = vec![];
        inject_summary_into_messages(&mut messages, "task", "summary", 2);
        assert!(messages.is_empty());
    }
}
