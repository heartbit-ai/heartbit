//! Built-in agent tool implementations — filesystem I/O, web, code execution, search, and more.

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
#[cfg(feature = "ghost-domain-config")]
pub(crate) mod twitter_post;
mod webfetch;
mod websearch;
mod write;

use std::path::PathBuf;
use std::sync::Arc;

use crate::tool::Tool;

/// Check if a path matches any protected pattern.
///
/// SECURITY (F-FS-11): the input path is normalised first
/// (`crate::workspace::normalize_path`) so trivial variants like
/// `/home/user//.ssh/key` or `/home/user/./.ssh/key` cannot bypass a
/// `~/.ssh` protection. Extension match is case-insensitive (ASCII)
/// since `secret.ENV` and `secret.env` are the same file on
/// HFS+/APFS/NTFS.
fn is_protected(path: &std::path::Path, protected: &[PathBuf]) -> bool {
    let normalized = crate::workspace::normalize_path(path);
    for pp in protected {
        if normalized.starts_with(pp) || normalized == *pp {
            return true;
        }
        if let Some(pattern) = pp.to_str()
            && let Some(pat_ext) = pattern.strip_prefix("*.")
            && let Some(ext) = normalized.extension().and_then(|e| e.to_str())
            && ext.eq_ignore_ascii_case(pat_ext)
        {
            return true;
        }
    }
    false
}

/// Write `bytes` to `path`, refusing to follow symlinks at the final
/// component (Unix). On non-Unix platforms, falls back to plain `tokio::fs::write`.
///
/// SECURITY (F-FS-1): combined with `CorePathPolicy::check_path_for_create`,
/// this eliminates the TOCTOU window where a parallel tool call could have
/// replaced the target path with a symlink between the policy check and the
/// open syscall.
pub(crate) async fn write_no_follow(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let path_owned = path.to_path_buf();
        let bytes = bytes.to_vec();
        tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&path_owned)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            Ok(())
        })
        .await
        .map_err(|e| std::io::Error::other(format!("spawn_blocking failed: {e}")))?
    }
    #[cfg(not(unix))]
    {
        // No equivalent of O_NOFOLLOW in the std/tokio stable API on Windows.
        // The `CorePathPolicy::check_path_for_create` parent-canonicalize check
        // still applies; combined with the absence of inotify-style TOCTOU
        // primitives this is the best we can do portably here.
        tokio::fs::write(path, bytes).await
    }
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

/// Round `target` down to the nearest UTF-8 character boundary in `text`.
///
/// Returns the largest position `≤ target.min(text.len())` that lies on a char
/// boundary, so `&text[..pos]` is always a valid UTF-8 slice.
pub fn floor_char_boundary(text: &str, target: usize) -> usize {
    let mut pos = target.min(text.len());
    while pos > 0 && !text.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

pub use file_tracker::FileTracker;
pub use image_generate::ImageGenerateTool;
pub use question::{
    OnQuestion, Question, QuestionOption, QuestionRequest, QuestionResponse, QuestionTool,
};
pub use todo::{TodoPriority, TodoStatus, TodoStore};
#[cfg(feature = "ghost-domain-config")]
pub use twitter_post::TwitterCredentials;
pub use webfetch::WebFetchTool;
pub use websearch::WebSearchTool;

/// Risk classification for builtin tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolRisk {
    /// Safe tools: always available.
    Safe,
    /// Dangerous tools: bash. Disabled by default in daemon mode.
    Dangerous,
}

/// Configuration for creating built-in tools.
#[non_exhaustive]
pub struct BuiltinToolsConfig {
    /// Shared mtime-based read-before-write tracker.
    pub file_tracker: Arc<FileTracker>,
    /// Shared todo-list store for the `todo` tool.
    pub todo_store: Arc<TodoStore>,
    /// Optional callback that resolves structured questions raised by the agent.
    pub on_question: Option<Arc<OnQuestion>>,
    /// Optional workspace root; tools restrict path access to descendants when set.
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
    /// X/Twitter credentials for the `twitter_post` builtin tool (per-tenant).
    #[cfg(feature = "ghost-domain-config")]
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
    /// Explicit, highest-precedence skill directories given to the `skill` tool
    /// (each holding `<name>/SKILL.md`). These are searched before the
    /// conventional `.opencode/.claude/skills` walk and are the SAME dirs the
    /// Level-1 catalog is built from, so the catalog never advertises a skill the
    /// tool cannot load. Empty (default) = conventional discovery only.
    pub skill_dirs: Vec<PathBuf>,
}

/// Sensible default `protected_paths` patterns for filesystem builtins.
///
/// SECURITY (F-FS-9): when no workspace is configured, file builtins
/// otherwise accept arbitrary absolute paths under the process's identity.
/// The default `protected_paths` is now populated with widely-recognised
/// secret-bearing files and directories. Operators can override via
/// `BuiltinToolsConfig::protected_paths` (replaces the default).
pub fn default_protected_paths() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = vec![
        // Patterns (matched by glob in is_protected). The leading `*.` form
        // is recognised by the existing matcher.
        PathBuf::from("*.env"),
        PathBuf::from("*.pem"),
        PathBuf::from("*.key"),
        PathBuf::from("*.p12"),
        PathBuf::from("*.pfx"),
        PathBuf::from("*.kdbx"),
        // Common system / process secret containers.
        PathBuf::from("/etc/shadow"),
        PathBuf::from("/etc/sudoers"),
        PathBuf::from("/proc/self/environ"),
    ];
    if let Some(home) = std::env::var_os("HOME") {
        let h = PathBuf::from(home);
        v.push(h.join(".ssh"));
        v.push(h.join(".aws"));
        v.push(h.join(".gnupg"));
        v.push(h.join(".config").join("heartbit"));
        v.push(h.join(".docker").join("config.json"));
        v.push(h.join(".netrc"));
    }
    v
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
            // SECURITY (F-FS-9): default protected paths populated.
            protected_paths: default_protected_paths(),
            #[cfg(all(target_os = "linux", feature = "sandbox"))]
            sandbox_policy: None,
            #[cfg(feature = "ghost-domain-config")]
            twitter_credentials: None,
            allowlist: None,
            path_policy: None,
            skill_dirs: Vec::new(),
        }
    }
}

/// Create all built-in tools with shared state.
pub fn builtin_tools(config: BuiltinToolsConfig) -> Vec<Arc<dyn Tool>> {
    let ws = config.workspace.map(|w| w.canonicalize().unwrap_or(w));
    let pp = Arc::new(config.protected_paths);
    let path_policy = config.path_policy;
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();

    macro_rules! maybe_policy {
        ($tool:expr) => {
            if let Some(ref pp) = path_policy {
                $tool.with_path_policy(Arc::clone(pp))
            } else {
                $tool
            }
        };
    }

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
                Arc::new(maybe_policy!(tool))
            }
            None => Arc::new(maybe_policy!(bash::BashTool::new())),
        };
        tools.push(bash_tool);
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
        // SECURITY (F-FS-4): apply the same `path_policy` to grep/glob/list
        // that read/write/edit/patch already use. Without this, an LLM with
        // no workspace configured can enumerate `/home` or grep `/etc` for
        // secrets while the path policy blocks the file builtins.
        Arc::new(maybe_policy!(grep::GrepTool::new(
            ws.clone(),
            Arc::clone(&pp)
        ))),
        Arc::new(maybe_policy!(glob::GlobTool::new(
            ws.clone(),
            Arc::clone(&pp)
        ))),
        Arc::new(maybe_policy!(list::ListTool::new(
            ws.clone(),
            Arc::clone(&pp)
        ))),
        Arc::new(maybe_policy!(patch::PatchTool::new(
            config.file_tracker.clone(),
            ws,
            Arc::clone(&pp),
        ))),
        Arc::new(webfetch::WebFetchTool::new()),
        Arc::new(websearch::WebSearchTool::new()),
        Arc::new(image_generate::ImageGenerateTool::new()),
        Arc::new(tts::TtsTool::new()),
        Arc::new(skill::SkillTool::with_dirs(config.skill_dirs.clone())),
    ]);

    let todo_tools = todo::todo_tools(config.todo_store);
    tools.extend(todo_tools);

    if let Some(on_question) = config.on_question {
        tools.push(Arc::new(question::QuestionTool::new(on_question)));
    }

    #[cfg(feature = "ghost-domain-config")]
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

    #[cfg(feature = "ghost-domain-config")]
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

    #[cfg(feature = "ghost-domain-config")]
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
