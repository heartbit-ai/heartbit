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

use std::path::PathBuf;
use std::sync::Arc;

use crate::tool::Tool;

/// Resolve a file path: absolute paths pass through, relative paths are
/// resolved against the workspace root (or CWD if no workspace is set).
pub(crate) fn resolve_path(path: &str, workspace: Option<&std::path::Path>) -> PathBuf {
    let p = std::path::Path::new(path);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    match workspace {
        Some(ws) => ws.join(p),
        None => p.to_path_buf(),
    }
}

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
pub use todo::{TodoPriority, TodoStatus, TodoStore};

/// Configuration for creating built-in tools.
pub struct BuiltinToolsConfig {
    /// Shared file tracker for read-before-write guard.
    pub file_tracker: Arc<FileTracker>,
    /// Shared todo store for session-scoped task tracking.
    pub todo_store: Arc<TodoStore>,
    /// Optional callback for structured questions to the user.
    pub on_question: Option<Arc<OnQuestion>>,
    /// Optional workspace root directory. When set, file tools resolve
    /// relative paths against this directory and BashTool starts here.
    pub workspace: Option<PathBuf>,
    /// Optional persistent daemon todo store. When set, the `todo_manage`
    /// tool is added to the built-in tools for managing the daemon's
    /// persistent task list.
    #[cfg(feature = "daemon")]
    pub daemon_todo_store: Option<Arc<crate::daemon::todo::FileTodoStore>>,
}

impl Default for BuiltinToolsConfig {
    fn default() -> Self {
        Self {
            file_tracker: Arc::new(FileTracker::new()),
            todo_store: Arc::new(TodoStore::new()),
            on_question: None,
            workspace: None,
            #[cfg(feature = "daemon")]
            daemon_todo_store: None,
        }
    }
}

/// Create all built-in tools with shared state.
///
/// Returns a `Vec<Arc<dyn Tool>>` ready to pass to `AgentRunnerBuilder::tools()`.
pub fn builtin_tools(config: BuiltinToolsConfig) -> Vec<Arc<dyn Tool>> {
    let ws = config.workspace.clone();
    let bash_tool: Arc<dyn Tool> = match &ws {
        Some(path) => Arc::new(bash::BashTool::with_workspace(path.clone())),
        None => Arc::new(bash::BashTool::new()),
    };
    let mut tools: Vec<Arc<dyn Tool>> = vec![
        bash_tool,
        Arc::new(read::ReadTool::new(config.file_tracker.clone(), ws.clone())),
        Arc::new(write::WriteTool::new(
            config.file_tracker.clone(),
            ws.clone(),
        )),
        Arc::new(edit::EditTool::new(config.file_tracker.clone(), ws.clone())),
        Arc::new(grep::GrepTool::new(ws.clone())),
        Arc::new(glob::GlobTool::new(ws.clone())),
        Arc::new(list::ListTool::new(ws.clone())),
        Arc::new(patch::PatchTool::new(config.file_tracker.clone(), ws)),
        Arc::new(webfetch::WebFetchTool::new()),
        Arc::new(websearch::WebSearchTool::new()),
        Arc::new(skill::SkillTool::new()),
    ];

    let todo_tools = todo::todo_tools(config.todo_store);
    tools.extend(todo_tools);

    if let Some(on_question) = config.on_question {
        tools.push(Arc::new(question::QuestionTool::new(on_question)));
    }

    #[cfg(feature = "daemon")]
    if let Some(daemon_store) = config.daemon_todo_store {
        tools.push(Arc::new(crate::daemon::todo::TodoManageTool::new(
            daemon_store,
        )));
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

    #[test]
    fn resolve_path_absolute_passthrough_with_workspace() {
        let ws = Some(std::path::Path::new("/workspace"));
        let result = resolve_path("/absolute/path", ws);
        assert_eq!(result, PathBuf::from("/absolute/path"));
    }

    #[test]
    fn resolve_path_absolute_passthrough_without_workspace() {
        let result = resolve_path("/absolute/path", None);
        assert_eq!(result, PathBuf::from("/absolute/path"));
    }

    #[test]
    fn resolve_path_relative_with_workspace() {
        let ws = Some(std::path::Path::new("/workspace"));
        let result = resolve_path("notes.md", ws);
        assert_eq!(result, PathBuf::from("/workspace/notes.md"));
    }

    #[test]
    fn resolve_path_relative_nested_with_workspace() {
        let ws = Some(std::path::Path::new("/workspace"));
        let result = resolve_path("subdir/notes.md", ws);
        assert_eq!(result, PathBuf::from("/workspace/subdir/notes.md"));
    }

    #[test]
    fn resolve_path_relative_without_workspace() {
        let result = resolve_path("notes.md", None);
        assert_eq!(result, PathBuf::from("notes.md"));
    }

    #[test]
    fn builtin_tools_count() {
        let tools = builtin_tools(BuiltinToolsConfig::default());
        // 8 file tools + webfetch + websearch + skill + 2 todo = 13 (no question without callback)
        assert_eq!(tools.len(), 13);
    }

    #[test]
    fn builtin_tools_with_question_callback() {
        let config = BuiltinToolsConfig {
            on_question: Some(Arc::new(|_| {
                Box::pin(async { Ok(QuestionResponse { answers: vec![] }) })
            })),
            ..Default::default()
        };
        let tools = builtin_tools(config);
        assert_eq!(tools.len(), 14); // 13 + question tool
    }

    #[cfg(feature = "daemon")]
    #[test]
    fn builtin_tools_with_daemon_todo_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::daemon::todo::FileTodoStore::new(dir.path()).unwrap());
        let config = BuiltinToolsConfig {
            daemon_todo_store: Some(store),
            ..Default::default()
        };
        let tools = builtin_tools(config);
        assert_eq!(tools.len(), 14); // 13 + todo_manage tool
        assert!(tools.iter().any(|t| t.definition().name == "todo_manage"));
    }
}
