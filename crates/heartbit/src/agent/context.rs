use crate::llm::types::{CompletionRequest, Message, ToolDefinition, ToolResult};

/// Conversation context for an agent run.
pub struct AgentContext {
    system: String,
    messages: Vec<Message>,
    tools: Vec<ToolDefinition>,
    max_turns: usize,
    current_turn: usize,
    model: String,
}

impl AgentContext {
    pub fn new(
        system: impl Into<String>,
        task: impl Into<String>,
        tools: Vec<ToolDefinition>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            system: system.into(),
            messages: vec![Message::user(task)],
            tools,
            max_turns: 10,
            current_turn: 0,
            model: model.into(),
        }
    }

    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
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

    pub fn to_request(&self) -> CompletionRequest {
        CompletionRequest {
            model: self.model.clone(),
            system: self.system.clone(),
            messages: self.messages.clone(),
            tools: self.tools.clone(),
            max_tokens: 4096,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn new_context_has_user_message() {
        let ctx = AgentContext::new("system", "do something", vec![], "model");
        let req = ctx.to_request();

        assert_eq!(req.system, "system");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, crate::llm::types::Role::User);
    }

    #[test]
    fn with_max_turns_overrides_default() {
        let ctx = AgentContext::new("sys", "task", vec![], "m").with_max_turns(5);
        assert_eq!(ctx.max_turns(), 5);
    }

    #[test]
    fn turn_tracking() {
        let mut ctx = AgentContext::new("sys", "task", vec![], "m");
        assert_eq!(ctx.current_turn(), 0);
        ctx.increment_turn();
        assert_eq!(ctx.current_turn(), 1);
    }

    #[test]
    fn add_tool_results_creates_user_message() {
        let mut ctx = AgentContext::new("sys", "task", vec![], "m");
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
        let ctx = AgentContext::new("sys", "task", tools, "m");
        let req = ctx.to_request();
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0].name, "search");
    }
}
