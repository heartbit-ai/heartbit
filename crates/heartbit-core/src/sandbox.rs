//! Path-level sandbox policy shared across filesystem-touching builtins.

use std::path::{Path, PathBuf};

use crate::error::Error;

/// Path-level policy for filesystem-touching tools.
///
/// Lives in heartbit-core (not in the umbrella) so all filesystem
/// builtins (bash, patch, edit, write, read) can share enforcement.
/// The umbrella's `SandboxPolicy` (landlock-backed on Linux) will
/// compose a `CorePathPolicy` for the path-allowlist piece. Until
/// Task 5 lands, the two are independent.
#[derive(Debug, Clone)]
pub struct CorePathPolicy {
    allowed_dirs: Vec<PathBuf>,
    deny_globs: Vec<glob::Pattern>,
}

impl CorePathPolicy {
    pub fn builder() -> CorePathPolicyBuilder {
        CorePathPolicyBuilder::default()
    }

    /// Returns `Ok(())` if `path` is allowed, `Err(Error::Sandbox(...))` otherwise.
    /// Canonicalizes the input so symlinks pointing outside `allowed_dirs`
    /// are rejected.
    pub fn check_path(&self, path: &Path) -> Result<(), Error> {
        let canonical = path
            .canonicalize()
            .map_err(|e| Error::Sandbox(format!("canonicalize {}: {e}", path.display())))?;

        let allowed = self
            .allowed_dirs
            .iter()
            .any(|root| canonical.starts_with(root));
        if !allowed {
            return Err(Error::Sandbox(format!(
                "path {} not under any allowed directory",
                canonical.display()
            )));
        }

        for pat in &self.deny_globs {
            if pat.matches_path(&canonical) {
                return Err(Error::Sandbox(format!(
                    "path {} matches deny pattern {}",
                    canonical.display(),
                    pat.as_str()
                )));
            }
        }

        Ok(())
    }
}

#[derive(Default, Debug)]
pub struct CorePathPolicyBuilder {
    allowed_dirs: Vec<PathBuf>,
    deny_globs: Vec<String>,
}

impl CorePathPolicyBuilder {
    /// Allow filesystem operations under `dir`. The directory is
    /// canonicalized at `build()` time; passing a path that doesn't
    /// exist or that the process can't resolve causes `build()` to
    /// return `Err(Error::Sandbox(...))`.
    pub fn allow_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.allowed_dirs.push(dir.as_ref().to_path_buf());
        self
    }

    /// Deny any path matching this glob even if it falls under an allowed dir.
    pub fn deny_glob(mut self, pat: impl Into<String>) -> Self {
        self.deny_globs.push(pat.into());
        self
    }

    pub fn build(self) -> Result<CorePathPolicy, Error> {
        let allowed_dirs = self
            .allowed_dirs
            .into_iter()
            .map(|p| {
                p.canonicalize()
                    .map_err(|e| Error::Sandbox(format!("allow_dir {}: {e}", p.display())))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let deny_globs = self
            .deny_globs
            .into_iter()
            .map(|p| {
                glob::Pattern::new(&p)
                    .map_err(|e| Error::Sandbox(format!("invalid deny glob {p}: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(CorePathPolicy {
            allowed_dirs,
            deny_globs,
        })
    }
}

#[cfg(all(target_os = "linux", feature = "sandbox"))]
pub use landlock_sandbox::SandboxPolicy;

#[cfg(all(target_os = "linux", feature = "sandbox"))]
mod landlock_sandbox {
    use std::io;
    use std::path::PathBuf;
    use std::sync::Arc;

    use landlock::{
        ABI, Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr,
    };

    use super::CorePathPolicy;
    use crate::error::Error;

    /// Filesystem sandbox policy applied to bash subprocess via `pre_exec`.
    ///
    /// Composes a `CorePathPolicy` (path allowlist + glob denylist, shared with
    /// non-bash filesystem tools) plus the kernel-level Landlock enforcement.
    #[derive(Debug, Clone)]
    pub struct SandboxPolicy {
        /// Application-level path policy. Shared with read/write/edit/patch tools.
        path_policy: Arc<CorePathPolicy>,
        /// Paths with read-only access (Landlock layer).
        pub read_paths: Vec<PathBuf>,
        /// Paths with read-write access (Landlock layer).
        pub write_paths: Vec<PathBuf>,
    }

    impl SandboxPolicy {
        /// Default policy: R/W on workspace, read-only on system dirs.
        pub fn workspace_only(workspace: &std::path::Path) -> Self {
            let read_paths = vec![
                PathBuf::from("/usr"),
                PathBuf::from("/lib"),
                PathBuf::from("/lib64"),
                PathBuf::from("/bin"),
                PathBuf::from("/etc"),
                PathBuf::from("/tmp"),
                workspace.to_path_buf(),
            ];
            let write_paths = vec![workspace.to_path_buf(), PathBuf::from("/tmp")];
            // Build a corresponding CorePathPolicy (best-effort: skip missing dirs).
            let mut builder = CorePathPolicy::builder();
            for p in read_paths.iter().chain(write_paths.iter()) {
                if p.exists() {
                    builder = builder.allow_dir(p);
                }
            }
            let path_policy = Arc::new(builder.build().unwrap_or_else(|e| {
                // workspace_only registers only existing directories and zero deny globs,
                // so CorePathPolicyBuilder::build() cannot fail here in practice. If it
                // does, panic loudly rather than silently substituting an empty policy
                // (which would leave the Landlock layer permissive but the path-policy
                // layer denying everything — a split-brain).
                unreachable!(
                    "CorePathPolicy build failed in workspace_only despite filtered inputs: {e}"
                )
            }));
            Self {
                path_policy,
                read_paths,
                write_paths,
            }
        }

        /// Build a SandboxPolicy from an externally-constructed `CorePathPolicy`.
        ///
        /// Used when one path policy is shared across multiple tools (bash + write +
        /// edit + ...). The Landlock layer (`read_paths` / `write_paths`) is
        /// initialized **empty**: this is intentional, but means callers that want
        /// kernel-level enforcement on top of the path policy MUST populate
        /// `read_paths` / `write_paths` before calling `into_pre_exec`. Otherwise
        /// the Landlock ruleset locks the subprocess out of all filesystem access.
        ///
        /// For the common "workspace" case, prefer `SandboxPolicy::workspace_only`
        /// which derives both layers from a single `&Path`.
        pub fn from_path_policy(path_policy: Arc<CorePathPolicy>) -> Self {
            Self {
                path_policy,
                read_paths: Vec::new(),
                write_paths: Vec::new(),
            }
        }

        /// Expose the inner `CorePathPolicy` so callers can pass it to non-bash
        /// filesystem tools that take `Arc<CorePathPolicy>`.
        pub fn path_policy(&self) -> Arc<CorePathPolicy> {
            self.path_policy.clone()
        }

        /// Create a `pre_exec` closure that applies Landlock rules.
        pub fn into_pre_exec(self) -> Result<impl FnMut() -> io::Result<()>, Error> {
            debug_assert!(
                !self.read_paths.is_empty() || !self.write_paths.is_empty(),
                "SandboxPolicy::into_pre_exec called with empty read_paths AND write_paths; \
                 the resulting Landlock ruleset would lock the subprocess out of all \
                 filesystem access. Use workspace_only() or populate read_paths/write_paths \
                 before calling into_pre_exec on a from_path_policy()-built sandbox."
            );

            let abi = ABI::V5;

            let read_access = AccessFs::from_read(abi);
            let write_access = AccessFs::from_all(abi);

            let read_fds: Vec<_> = self
                .read_paths
                .iter()
                .filter_map(|p| PathFd::new(p).ok())
                .collect();

            let write_fds: Vec<_> = self
                .write_paths
                .iter()
                .filter_map(|p| PathFd::new(p).ok())
                .collect();

            Ok(move || {
                let mut ruleset = Ruleset::default()
                    .handle_access(write_access)
                    .map_err(|e| io::Error::other(e.to_string()))?
                    .create()
                    .map_err(|e| io::Error::other(e.to_string()))?;

                for fd in &read_fds {
                    ruleset = ruleset
                        .add_rule(PathBeneath::new(fd, read_access))
                        .map_err(|e| io::Error::other(e.to_string()))?;
                }

                for fd in &write_fds {
                    ruleset = ruleset
                        .add_rule(PathBeneath::new(fd, write_access))
                        .map_err(|e| io::Error::other(e.to_string()))?;
                }

                ruleset
                    .restrict_self()
                    .map_err(|e| io::Error::other(e.to_string()))?;

                Ok(())
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn workspace_only_includes_system_dirs() {
            let dir = tempfile::tempdir().unwrap();
            let policy = SandboxPolicy::workspace_only(dir.path());
            assert!(policy.read_paths.contains(&PathBuf::from("/usr")));
            assert!(policy.read_paths.contains(&PathBuf::from("/bin")));
            assert!(policy.read_paths.contains(&PathBuf::from("/etc")));
        }

        #[test]
        fn into_pre_exec_succeeds_on_workspace() {
            let dir = tempfile::tempdir().unwrap();
            let policy = SandboxPolicy::workspace_only(dir.path());
            let result = policy.into_pre_exec();
            assert!(result.is_ok());
        }

        #[test]
        fn workspace_only_includes_workspace_in_read_and_write() {
            let dir = tempfile::tempdir().unwrap();
            let policy = SandboxPolicy::workspace_only(dir.path());
            assert!(policy.read_paths.contains(&dir.path().to_path_buf()));
            assert!(policy.write_paths.contains(&dir.path().to_path_buf()));
        }

        #[test]
        fn from_path_policy_exposes_inner_policy() {
            let path_policy = Arc::new(
                CorePathPolicy::builder()
                    .allow_dir(std::env::temp_dir())
                    .build()
                    .unwrap(),
            );
            let sandbox = SandboxPolicy::from_path_policy(path_policy.clone());
            assert!(Arc::ptr_eq(&path_policy, &sandbox.path_policy()));
            assert!(sandbox.read_paths.is_empty());
            assert!(sandbox.write_paths.is_empty());
        }

        #[test]
        fn workspace_only_populates_path_policy() {
            let dir = tempfile::tempdir().unwrap();
            let policy = SandboxPolicy::workspace_only(dir.path());
            // The inner CorePathPolicy should accept paths under dir
            let inner = policy.path_policy();
            let file = dir.path().join("ok.txt");
            std::fs::write(&file, b"x").unwrap();
            assert!(inner.check_path(&file).is_ok());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn allows_path_under_allowed_dir() {
        let root = tmp();
        let file = root.path().join("ok.txt");
        fs::write(&file, b"x").unwrap();
        let policy = CorePathPolicy::builder()
            .allow_dir(root.path())
            .build()
            .unwrap();
        assert!(policy.check_path(&file).is_ok());
    }

    #[test]
    fn denies_path_outside_allowed_dirs() {
        let root = tmp();
        let policy = CorePathPolicy::builder()
            .allow_dir(root.path())
            .build()
            .unwrap();
        let bad_dir = tmp();
        let bad = bad_dir.path().join("x.txt");
        fs::write(&bad, b"x").unwrap();
        let err = policy.check_path(&bad).unwrap_err();
        assert!(matches!(err, Error::Sandbox(_)));
    }

    #[test]
    fn denies_glob_match_inside_allowed_dir() {
        let root = tmp();
        let dotenv = root.path().join(".env");
        fs::write(&dotenv, b"x").unwrap();
        let policy = CorePathPolicy::builder()
            .allow_dir(root.path())
            .deny_glob("**/.env")
            .build()
            .unwrap();
        let err = policy.check_path(&dotenv).unwrap_err();
        assert!(matches!(err, Error::Sandbox(_)));
    }

    #[test]
    fn empty_allowlist_denies_everything() {
        let policy = CorePathPolicy::builder().build().unwrap();
        let some_path = std::env::temp_dir();
        let err = policy.check_path(&some_path).unwrap_err();
        assert!(matches!(err, Error::Sandbox(_)));
    }

    #[test]
    fn invalid_glob_pattern_returns_error() {
        let result = CorePathPolicy::builder().deny_glob("[unclosed").build();
        assert!(result.is_err());
    }

    #[test]
    fn allow_dir_with_nonexistent_path_fails_at_build() {
        let bogus = std::env::temp_dir().join(format!("does-not-exist-{}", uuid::Uuid::new_v4()));
        let result = CorePathPolicy::builder().allow_dir(&bogus).build();
        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[test]
    fn denies_symlink_pointing_outside_allowed_dir() {
        use std::os::unix::fs::symlink;
        let allowed = tmp();
        let outside = tmp();
        let outside_file = outside.path().join("secret.txt");
        fs::write(&outside_file, b"secret").unwrap();

        // Create a symlink inside the allowed dir that points OUTSIDE.
        let link = allowed.path().join("link.txt");
        symlink(&outside_file, &link).unwrap();

        let policy = CorePathPolicy::builder()
            .allow_dir(allowed.path())
            .build()
            .unwrap();
        let err = policy.check_path(&link).unwrap_err();
        assert!(matches!(err, Error::Sandbox(_)));
    }
}
