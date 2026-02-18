use crate::llm::types::{CompletionRequest, Message, ToolDefinition, ToolResult};

/// Conversation context for an agent run.
pub struct AgentContext {
    system: String,
    messages: Vec<Message>,
    tools: Vec<ToolDefinition>,
    max_turns: usize,
    max_tokens: u32,
    current_turn: usize,
}

impl AgentContext {
    pub fn new(
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
        }
    }

    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn current_turn(&self) -> usize {
        self.current_turn
    }

    pub fn max_turns(&self) -> usize {
        self.max_turns
    }

    pub fn increment_turn(&mut self) {
        self.current_turn += 1;
    }

    pub fn add_assistant_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn add_tool_results(&mut self, results: Vec<ToolResult>) {
        self.messages.push(Message::tool_results(results));
    }

    /// Get the text from the last assistant message (avoids re-cloning the response).
    pub fn last_assistant_text(&self) -> Option<String> {
        self.messages.iter().rev().find_map(|m| {
            if m.role == crate::llm::types::Role::Assistant {
                let text: String = m
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        crate::llm::types::ContentBlock::Text { text } => Some(text.as_str()),
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

    pub fn to_request(&self) -> CompletionRequest {
        CompletionRequest {
            system: self.system.clone(),
            messages: self.messages.clone(),
            tools: self.tools.clone(),
            max_tokens: self.max_tokens,
        }
    }
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
        assert_eq!(req.messages[0].role, crate::llm::types::Role::User);
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
        assert_eq!(req.messages[1].role, crate::llm::types::Role::User);
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
}
