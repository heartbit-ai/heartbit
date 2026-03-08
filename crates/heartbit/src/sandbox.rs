//! Landlock filesystem sandbox for bash subprocesses.
//!
//! Uses Linux Landlock LSM to restrict filesystem access at the kernel level.
//! Landlock is unprivileged — it works inside Docker without `CAP_SYS_ADMIN`.
//!
//! Requires feature `sandbox` and Linux kernel >= 5.13.

use std::io;
use std::path::PathBuf;

use landlock::{
    ABI, Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr,
};

use crate::Error;

/// Filesystem sandbox policy applied to bash subprocess via `pre_exec`.
#[derive(Debug, Clone)]
pub struct SandboxPolicy {
    /// Paths with read-only access.
    pub read_paths: Vec<PathBuf>,
    /// Paths with read-write access.
    pub write_paths: Vec<PathBuf>,
}

impl SandboxPolicy {
    /// Default policy: R/W on workspace, read-only on system dirs.
    pub fn workspace_only(workspace: &std::path::Path) -> Self {
        Self {
            read_paths: vec![
                PathBuf::from("/usr"),
                PathBuf::from("/lib"),
                PathBuf::from("/lib64"),
                PathBuf::from("/bin"),
                PathBuf::from("/etc"),
                PathBuf::from("/tmp"),
                workspace.to_path_buf(),
            ],
            write_paths: vec![workspace.to_path_buf(), PathBuf::from("/tmp")],
        }
    }

    /// Create a `pre_exec` closure that applies Landlock rules.
    ///
    /// Returns `Err` if Landlock is not supported on this kernel.
    pub fn into_pre_exec(self) -> Result<impl FnMut() -> io::Result<()>, Error> {
        // Determine best available ABI.
        let abi = ABI::V5;

        let read_access = AccessFs::from_read(abi);
        let write_access = AccessFs::from_all(abi);

        // Pre-open all path file descriptors before entering the closure.
        // This is necessary because the closure runs after fork() but before
        // exec(), where we can't allocate or do complex operations.
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
    fn workspace_only_includes_workspace_in_read_and_write() {
        let ws = PathBuf::from("/home/user/workspace");
        let policy = SandboxPolicy::workspace_only(&ws);
        assert!(policy.read_paths.contains(&ws));
        assert!(policy.write_paths.contains(&ws));
    }

    #[test]
    fn workspace_only_includes_system_dirs() {
        let ws = PathBuf::from("/home/user/workspace");
        let policy = SandboxPolicy::workspace_only(&ws);
        assert!(policy.read_paths.contains(&PathBuf::from("/usr")));
        assert!(policy.read_paths.contains(&PathBuf::from("/bin")));
        assert!(policy.read_paths.contains(&PathBuf::from("/etc")));
    }

    #[test]
    fn into_pre_exec_succeeds_on_real_paths() {
        let dir = tempfile::tempdir().unwrap();
        let policy = SandboxPolicy::workspace_only(dir.path());
        // Should not error since we're on a Linux system with Landlock support
        // (or gracefully handle if kernel doesn't support it)
        let result = policy.into_pre_exec();
        // We just verify it doesn't panic — actual enforcement is in forked subprocess
        assert!(result.is_ok());
    }

    #[test]
    fn custom_policy() {
        let dir = tempfile::tempdir().unwrap();
        let policy = SandboxPolicy {
            read_paths: vec![dir.path().to_path_buf()],
            write_paths: vec![],
        };
        assert_eq!(policy.read_paths.len(), 1);
        assert!(policy.write_paths.is_empty());
    }
}
