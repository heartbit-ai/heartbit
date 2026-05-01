mod bash;
mod edit;
mod file_tracker;
mod glob;
mod grep;
mod image_generate;
mod list;
mod patch;
mod question;
mod read;
mod skill;
mod todo;
mod tts;
pub(crate) mod twitter_post;
mod webfetch;
mod websearch;
mod write;

use std::path::PathBuf;
use std::sync::Arc;

use crate::tool::Tool;

/// Check if a path matches any protected pattern.
fn is_protected(path: &std::path::Path, protected: &[PathBuf]) -> bool {
    for pp in protected {
        if path.starts_with(pp) || path == pp {
            return true;
        }
        if let Some(pattern) = pp.to_str()
            && let Some(pat_ext) = pattern.strip_prefix("*.")
            && let Some(ext) = path.extension().and_then(|e| e.to_str())
            && ext == pat_ext
        {
            return true;
        }
    }
    false
}

/// Resolve a file path with workspace jail enforcement and protected path checks.
pub(crate) fn resolve_path(
    path: &str,
    workspace: Option<&std::path::Path>,
    protected_paths: &[PathBuf],
) -> Result<PathBuf, String> {
    let p = std::path::Path::new(path);

    match workspace {
        Some(ws) => {
            if p.is_absolute() {
                return Err(format!(
                    "Absolute paths are not allowed when workspace is set. \
                     Use a relative path instead of '{path}'."
                ));
            }
            let candidate = ws.join(p);
            let normalized = crate::workspace::normalize_path(&candidate);
            if !normalized.starts_with(ws) {
                return Err(format!(
                    "Path '{path}' escapes the workspace root ({}).",
                    ws.display()
                ));
            }
            if let Ok(canonical) = normalized.canonicalize()
                && !canonical.starts_with(ws)
            {
                return Err(format!(
                    "Path '{path}' resolves to {} which is outside the workspace.",
                    canonical.display()
                ));
            }
            if is_protected(&normalized, protected_paths) {
                return Err(format!("Access to '{path}' is denied (protected path)."));
            }
            Ok(normalized)
        }
        None => {
            let result = p.to_path_buf();
            if is_protected(&result, protected_paths) {
                return Err(format!("Access to '{path}' is denied (protected path)."));
            }
            Ok(result)
        }
    }
}

pub fn floor_char_boundary(text: &str, target: usize) -> usize {
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
pub use twitter_post::TwitterCredentials;

/// Risk classification for builtin tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolRisk {
    /// Safe tools: always available.
    Safe,
    /// Dangerous tools: bash. Disabled by default in daemon mode.
    Dangerous,
}

/// Configuration for creating built-in tools.
pub struct BuiltinToolsConfig {
    pub file_tracker: Arc<FileTracker>,
    pub todo_store: Arc<TodoStore>,
    pub on_question: Option<Arc<OnQuestion>>,
    pub workspace: Option<PathBuf>,
    /// Enable dangerous tools (e.g. bash). Default: false.
    pub dangerous_tools: bool,
    /// Environment variable policy for bash subprocesses.
    pub env_policy: crate::workspace::EnvPolicy,
    /// File path patterns to deny access to (e.g., `*.env`, `*.pem`).
    pub protected_paths: Vec<PathBuf>,
    /// Landlock filesystem sandbox policy for bash (Linux only).
    #[cfg(all(target_os = "linux", feature = "sandbox"))]
    pub sandbox_policy: Option<crate::sandbox::SandboxPolicy>,
    #[cfg(feature = "daemon")]
    pub daemon_todo_store: Option<Arc<crate::daemon::todo::FileTodoStore>>,
    /// X/Twitter credentials for the `twitter_post` builtin tool (per-tenant).
    pub twitter_credentials: Option<TwitterCredentials>,
    /// Optional allowlist of builtin tool names. When `Some`, only tools whose
    /// name appears in this list are returned. When `None`, all builtins are
    /// returned (backward compatible).
    pub allowlist: Option<Vec<String>>,
    /// Application-layer path policy applied to all filesystem builtins
    /// (bash, read, write, edit, patch). When set, `check_path` is called
    /// before any I/O, complementing the existing workspace + protected_paths
    /// mechanism and the Linux-only Landlock sandbox.
    pub path_policy: Option<Arc<crate::sandbox::CorePathPolicy>>,
}

impl Default for BuiltinToolsConfig {
    fn default() -> Self {
        Self {
            file_tracker: Arc::new(FileTracker::new()),
            todo_store: Arc::new(TodoStore::new()),
            on_question: None,
            workspace: None,
            dangerous_tools: false,
            env_policy: crate::workspace::EnvPolicy::Inherit,
            protected_paths: Vec::new(),
            #[cfg(all(target_os = "linux", feature = "sandbox"))]
            sandbox_policy: None,
            #[cfg(feature = "daemon")]
            daemon_todo_store: None,
            twitter_credentials: None,
            allowlist: None,
            path_policy: None,
        }
    }
}

/// Create all built-in tools with shared state.
pub fn builtin_tools(config: BuiltinToolsConfig) -> Vec<Arc<dyn Tool>> {
    let ws = config.workspace.map(|w| w.canonicalize().unwrap_or(w));
    let pp = Arc::new(config.protected_paths);
    let path_policy = config.path_policy;
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();

    if config.dangerous_tools {
        let bash_tool: Arc<dyn Tool> = match &ws {
            Some(path) => {
                let tool = bash::BashTool::with_sandbox(path.clone(), config.env_policy);
                #[cfg(all(target_os = "linux", feature = "sandbox"))]
                let tool = if let Some(policy) = config.sandbox_policy {
                    tool.with_sandbox_policy(policy)
                } else {
                    tool
                };
                let tool = if let Some(ref pp) = path_policy {
                    tool.with_path_policy(Arc::clone(pp))
                } else {
                    tool
                };
                Arc::new(tool)
            }
            None => {
                let tool = bash::BashTool::new();
                let tool = if let Some(ref pp) = path_policy {
                    tool.with_path_policy(Arc::clone(pp))
                } else {
                    tool
                };
                Arc::new(tool)
            }
        };
        tools.push(bash_tool);
    }

    macro_rules! maybe_policy {
        ($tool:expr) => {
            if let Some(ref pp) = path_policy {
                $tool.with_path_policy(Arc::clone(pp))
            } else {
                $tool
            }
        };
    }

    tools.extend([
        Arc::new(maybe_policy!(read::ReadTool::new(
            config.file_tracker.clone(),
            ws.clone(),
            Arc::clone(&pp),
        ))) as Arc<dyn Tool>,
        Arc::new(maybe_policy!(write::WriteTool::new(
            config.file_tracker.clone(),
            ws.clone(),
            Arc::clone(&pp),
        ))),
        Arc::new(maybe_policy!(edit::EditTool::new(
            config.file_tracker.clone(),
            ws.clone(),
            Arc::clone(&pp),
        ))),
        Arc::new(grep::GrepTool::new(ws.clone(), Arc::clone(&pp))),
        Arc::new(glob::GlobTool::new(ws.clone(), Arc::clone(&pp))),
        Arc::new(list::ListTool::new(ws.clone(), Arc::clone(&pp))),
        Arc::new(maybe_policy!(patch::PatchTool::new(
            config.file_tracker.clone(),
            ws,
            Arc::clone(&pp),
        ))),
        Arc::new(webfetch::WebFetchTool::new()),
        Arc::new(websearch::WebSearchTool::new()),
        Arc::new(image_generate::ImageGenerateTool::new()),
        Arc::new(tts::TtsTool::new()),
        Arc::new(skill::SkillTool::new()),
    ]);

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

    if let Some(creds) = config.twitter_credentials {
        tools.push(Arc::new(twitter_post::TwitterPostTool::new(creds)));
    }

    if let Some(ref allowed) = config.allowlist {
        let set: std::collections::HashSet<&str> = allowed.iter().map(|s| s.as_str()).collect();
        tools.retain(|t| set.contains(t.definition().name.as_str()));
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
        let s = "café";
        assert_eq!(s.len(), 5);
        assert_eq!(floor_char_boundary(s, 4), 3);
        assert_eq!(floor_char_boundary(s, 3), 3);
        assert_eq!(floor_char_boundary(s, 5), 5);
    }

    #[test]
    fn resolve_path_absolute_rejected_with_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path();
        let result = resolve_path("/absolute/path", Some(ws), &[]);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("Absolute paths are not allowed")
        );
    }

    #[test]
    fn resolve_path_absolute_passthrough_without_workspace() {
        let result = resolve_path("/absolute/path", None, &[]);
        assert_eq!(result.unwrap(), PathBuf::from("/absolute/path"));
    }

    #[test]
    fn resolve_path_relative_with_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().canonicalize().unwrap();
        let result = resolve_path("notes.md", Some(&ws), &[]);
        assert_eq!(result.unwrap(), ws.join("notes.md"));
    }

    #[test]
    fn resolve_path_relative_without_workspace() {
        let result = resolve_path("notes.md", None, &[]);
        assert_eq!(result.unwrap(), PathBuf::from("notes.md"));
    }

    #[test]
    fn resolve_path_traversal_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().canonicalize().unwrap();
        let result = resolve_path("../../etc/passwd", Some(&ws), &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("escapes the workspace"));
    }

    #[test]
    fn resolve_path_internal_dotdot_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().canonicalize().unwrap();
        let result = resolve_path("sub/../file.txt", Some(&ws), &[]);
        assert_eq!(result.unwrap(), ws.join("file.txt"));
    }

    #[test]
    fn resolve_path_boundary_dotdot_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().canonicalize().unwrap();
        let result = resolve_path("../escape", Some(&ws), &[]);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_path_symlink_escape_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().canonicalize().unwrap();
        let target = tempfile::tempdir().unwrap();
        std::fs::write(target.path().join("secret.txt"), "secret").unwrap();
        let link_path = ws.join("escape_link");
        #[cfg(unix)]
        std::os::unix::fs::symlink(target.path(), &link_path).unwrap();
        #[cfg(not(unix))]
        {
            return;
        }
        let result = resolve_path("escape_link/secret.txt", Some(&ws), &[]);
        assert!(
            result.is_err(),
            "symlink escape should be rejected: {:?}",
            result
        );
    }

    #[test]
    fn resolve_path_rejects_protected_extension() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().canonicalize().unwrap();
        std::fs::write(ws.join("secret.env"), "SECRET=value").unwrap();
        let protected = vec![PathBuf::from("*.env")];
        let result = resolve_path("secret.env", Some(&ws), &protected);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("protected"));
    }

    #[test]
    fn resolve_path_allows_non_protected() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().canonicalize().unwrap();
        let protected = vec![PathBuf::from("*.env")];
        let result = resolve_path("notes.md", Some(&ws), &protected);
        assert!(result.is_ok());
    }

    #[test]
    fn builtin_tools_excludes_bash_by_default() {
        let tools = builtin_tools(BuiltinToolsConfig::default());
        assert!(!tools.iter().any(|t| t.definition().name == "bash"));
        assert_eq!(tools.len(), 14);
    }

    #[test]
    fn builtin_tools_includes_bash_when_dangerous() {
        let config = BuiltinToolsConfig {
            dangerous_tools: true,
            ..Default::default()
        };
        let tools = builtin_tools(config);
        assert!(tools.iter().any(|t| t.definition().name == "bash"));
        assert_eq!(tools.len(), 15);
    }

    #[test]
    fn builtin_tools_with_question_callback() {
        let config = BuiltinToolsConfig {
            dangerous_tools: true,
            on_question: Some(Arc::new(|_| {
                Box::pin(async { Ok(QuestionResponse { answers: vec![] }) })
            })),
            ..Default::default()
        };
        let tools = builtin_tools(config);
        assert_eq!(tools.len(), 16);
    }

    #[cfg(feature = "daemon")]
    #[test]
    fn builtin_tools_with_daemon_todo_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::daemon::todo::FileTodoStore::new(dir.path()).unwrap());
        let config = BuiltinToolsConfig {
            dangerous_tools: true,
            daemon_todo_store: Some(store),
            ..Default::default()
        };
        let tools = builtin_tools(config);
        assert_eq!(tools.len(), 16);
        assert!(tools.iter().any(|t| t.definition().name == "todo_manage"));
    }

    #[test]
    fn builtin_tools_includes_twitter_when_credentials_present() {
        let config = BuiltinToolsConfig {
            twitter_credentials: Some(TwitterCredentials {
                consumer_key: "ck".into(),
                consumer_secret: "cs".into(),
                access_token: "at".into(),
                access_token_secret: "ats".into(),
            }),
            ..Default::default()
        };
        let tools = builtin_tools(config);
        assert_eq!(tools.len(), 15); // 14 base + twitter_post
        assert!(tools.iter().any(|t| t.definition().name == "twitter_post"));
    }

    #[test]
    fn builtin_tools_excludes_twitter_when_no_credentials() {
        let tools = builtin_tools(BuiltinToolsConfig::default());
        assert!(!tools.iter().any(|t| t.definition().name == "twitter_post"));
    }

    #[test]
    fn builtin_tools_with_allowlist() {
        let config = BuiltinToolsConfig {
            allowlist: Some(vec!["websearch".into(), "webfetch".into()]),
            ..Default::default()
        };
        let tools = builtin_tools(config);
        assert_eq!(tools.len(), 2);
        let names: Vec<String> = tools.iter().map(|t| t.definition().name.clone()).collect();
        assert!(names.contains(&"websearch".to_string()));
        assert!(names.contains(&"webfetch".to_string()));
    }

    #[test]
    fn builtin_tools_empty_allowlist() {
        let config = BuiltinToolsConfig {
            allowlist: Some(vec![]),
            ..Default::default()
        };
        let tools = builtin_tools(config);
        assert_eq!(tools.len(), 0);
    }

    #[test]
    fn builtin_tools_allowlist_none_returns_all() {
        let config = BuiltinToolsConfig {
            allowlist: None,
            ..Default::default()
        };
        let tools = builtin_tools(config);
        assert_eq!(tools.len(), 14);
    }

    #[test]
    fn builtin_tools_allowlist_bash_gated() {
        // Even if allowlist includes bash, dangerous_tools=false prevents it
        let config = BuiltinToolsConfig {
            dangerous_tools: false,
            allowlist: Some(vec!["bash".into()]),
            ..Default::default()
        };
        let tools = builtin_tools(config);
        assert_eq!(tools.len(), 0);
    }
}
