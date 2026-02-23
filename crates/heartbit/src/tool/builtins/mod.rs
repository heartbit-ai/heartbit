mod bash;
mod edit;
mod file_tracker;
mod glob;
mod grep;
mod list;
mod patch;
mod question;
mod read;
mod skill;
mod todo;
mod webfetch;
mod websearch;
mod write;

use std::sync::Arc;

use crate::tool::Tool;

/// Find the largest byte index that is a char boundary at or below `target`.
///
/// Used by multiple tools to truncate UTF-8 strings safely.
pub(crate) fn floor_char_boundary(text: &str, target: usize) -> usize {
    let mut pos = target.min(text.len());
    while pos > 0 && !text.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

pub use file_tracker::FileTracker;
pub use question::{
    OnQuestion, Question, QuestionOption, QuestionRequest, QuestionResponse, QuestionTool,
};
pub use todo::TodoStore;

/// Configuration for creating built-in tools.
pub struct BuiltinToolsConfig {
    /// Shared file tracker for read-before-write guard.
    pub file_tracker: Arc<FileTracker>,
    /// Shared todo store for session-scoped task tracking.
    pub todo_store: Arc<TodoStore>,
    /// Optional callback for structured questions to the user.
    pub on_question: Option<Arc<OnQuestion>>,
}

impl Default for BuiltinToolsConfig {
    fn default() -> Self {
        Self {
            file_tracker: Arc::new(FileTracker::new()),
            todo_store: Arc::new(TodoStore::new()),
            on_question: None,
        }
    }
}

/// Create all built-in tools with shared state.
///
/// Returns a `Vec<Arc<dyn Tool>>` ready to pass to `AgentRunnerBuilder::tools()`.
pub fn builtin_tools(config: BuiltinToolsConfig) -> Vec<Arc<dyn Tool>> {
    let mut tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(bash::BashTool::new()),
        Arc::new(read::ReadTool::new(config.file_tracker.clone())),
        Arc::new(write::WriteTool::new(config.file_tracker.clone())),
        Arc::new(edit::EditTool::new(config.file_tracker.clone())),
        Arc::new(grep::GrepTool::new()),
        Arc::new(glob::GlobTool::new()),
        Arc::new(list::ListTool::new()),
        Arc::new(patch::PatchTool::new(config.file_tracker.clone())),
        Arc::new(webfetch::WebFetchTool::new()),
        Arc::new(websearch::WebSearchTool::new()),
        Arc::new(skill::SkillTool::new()),
    ];

    let todo_tools = todo::todo_tools(config.todo_store);
    tools.extend(todo_tools);

    if let Some(on_question) = config.on_question {
        tools.push(Arc::new(question::QuestionTool::new(on_question)));
    }

    tools
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_char_boundary_ascii() {
        assert_eq!(floor_char_boundary("hello", 3), 3);
        assert_eq!(floor_char_boundary("hello", 10), 5);
        assert_eq!(floor_char_boundary("hello", 0), 0);
    }

    #[test]
    fn floor_char_boundary_multibyte() {
        // "café" is 5 bytes: c(1) a(1) f(1) é(2)
        let s = "café";
        assert_eq!(s.len(), 5);
        // Byte 4 is in the middle of 'é' (2-byte char starts at 3)
        assert_eq!(floor_char_boundary(s, 4), 3);
        assert_eq!(floor_char_boundary(s, 3), 3);
        assert_eq!(floor_char_boundary(s, 5), 5);
    }
}
