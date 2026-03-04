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

/// Resolve a file path with workspace jail enforcement.
///
/// When `workspace` is `Some(ws)`:
/// - **Absolute paths are rejected** (agents must use relative paths).
/// - Relative paths are joined to `ws`, normalized, and checked for containment.
/// - If the resolved path exists, symlinks are resolved via `canonicalize()` and
///   re-checked to prevent symlink escapes.
///
/// When `workspace` is `None` (CLI/standalone), behavior is unchanged: absolute
/// paths pass through, relative paths are returned as-is.
///
/// The `workspace` path should be pre-canonicalized (done once in `builtin_tools()`).
pub(crate) fn resolve_path(
    path: &str,
    workspace: Option<&std::path::Path>,
) -> Result<PathBuf, String> {
    let p = std::path::Path::new(path);

    match workspace {
        Some(ws) => {
            // Reject absolute paths — agents must stay inside workspace
            if p.is_absolute() {
                return Err(format!(
                    "Absolute paths are not allowed when workspace is set. \
                     Use a relative path instead of '{path}'."
                ));
            }

            // Normalize to resolve .. without touching the filesystem
            let candidate = ws.join(p);
            let normalized = crate::workspace::normalize_path(&candidate);

            if !normalized.starts_with(ws) {
                return Err(format!(
                    "Path '{path}' escapes the workspace root ({}).",
                    ws.display()
                ));
            }

            // Symlink check: canonicalize and re-verify against the workspace
            // root (which is already canonical from builtin_tools()).
            // TOCTOU note: a symlink could be swapped between this check and
            // the actual file open. Acceptable for agent tool jail; closing
            // it requires O_NOFOLLOW or OS-level namespaces.
            if let Ok(canonical) = normalized.canonicalize()
                && !canonical.starts_with(ws)
            {
                return Err(format!(
                    "Path '{path}' resolves to {} which is outside the workspace.",
                    canonical.display()
                ));
            }

            Ok(normalized)
        }
        None => {
            // No workspace — pass through unchanged
            Ok(p.to_path_buf())
        }
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
    // Pre-canonicalize workspace once so tools don't repeat canonicalize() on every call.
    let ws = config.workspace.map(|w| w.canonicalize().unwrap_or(w));
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
    fn resolve_path_absolute_rejected_with_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path();
        let result = resolve_path("/absolute/path", Some(ws));
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("Absolute paths are not allowed")
        );
    }

    #[test]
    fn resolve_path_absolute_passthrough_without_workspace() {
        let result = resolve_path("/absolute/path", None);
        assert_eq!(result.unwrap(), PathBuf::from("/absolute/path"));
    }

    #[test]
    fn resolve_path_relative_with_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().canonicalize().unwrap();
        let result = resolve_path("notes.md", Some(&ws));
        assert_eq!(result.unwrap(), ws.join("notes.md"));
    }

    #[test]
    fn resolve_path_relative_nested_with_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().canonicalize().unwrap();
        let result = resolve_path("subdir/notes.md", Some(&ws));
        assert_eq!(result.unwrap(), ws.join("subdir/notes.md"));
    }

    #[test]
    fn resolve_path_relative_without_workspace() {
        let result = resolve_path("notes.md", None);
        assert_eq!(result.unwrap(), PathBuf::from("notes.md"));
    }

    #[test]
    fn resolve_path_traversal_rejected_with_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().canonicalize().unwrap();
        let result = resolve_path("../../etc/passwd", Some(&ws));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("escapes the workspace"));
    }

    #[test]
    fn resolve_path_internal_dotdot_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().canonicalize().unwrap();
        let result = resolve_path("sub/../file.txt", Some(&ws));
        assert_eq!(result.unwrap(), ws.join("file.txt"));
    }

    #[test]
    fn resolve_path_boundary_dotdot_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().canonicalize().unwrap();
        // Exactly at the boundary: going up from root of workspace
        let result = resolve_path("../escape", Some(&ws));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("escapes the workspace"));
    }

    #[test]
    fn resolve_path_symlink_escape_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().canonicalize().unwrap();
        let target = tempfile::tempdir().unwrap();
        std::fs::write(target.path().join("secret.txt"), "secret").unwrap();

        // Create symlink inside workspace pointing outside
        let link_path = ws.join("escape_link");
        #[cfg(unix)]
        std::os::unix::fs::symlink(target.path(), &link_path).unwrap();
        #[cfg(not(unix))]
        {
            // Skip on non-unix
            return;
        }

        let result = resolve_path("escape_link/secret.txt", Some(&ws));
        assert!(
            result.is_err(),
            "symlink escape should be rejected: {:?}",
            result
        );
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
