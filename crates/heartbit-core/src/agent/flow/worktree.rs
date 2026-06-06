//! Git-worktree isolation for flow agent leaves (P5b).
//!
//! A leaf with `Isolation::Worktree` runs against ITS OWN git worktree of the
//! ctx workspace, so parallel file-mutating agents cannot trample each other.
//! Cleanup policy (the Claude Code semantic): a worktree left CLEAN is pruned;
//! a DIRTY one has its changes committed onto a `hb/<name>` branch (work is
//! never lost), then the directory is removed.
//!
//! Deviation from the parked design: this shells out to the `git` CLI
//! (`tokio::process`) instead of linking `git2` — zero new dependencies, no
//! `!Send` libgit2 handles across `.await`, no C build. Requires `git` on
//! `PATH` at runtime (the dev-workstation audience that wants worktrees has it).

use std::path::{Path, PathBuf};

use crate::error::Error;

/// What `cleanup` did with the worktree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Disposition {
    /// The worktree was clean — pruned without residue.
    Pruned,
    /// The worktree had changes — they were committed onto this branch in the
    /// parent repository, then the directory was removed.
    Kept {
        /// Branch name holding the preserved changes (`hb/<name>`).
        branch: String,
    },
}

/// One leaf's isolated worktree. Create with [`WorktreeGuard::create`], always
/// finish with [`WorktreeGuard::cleanup`] (a sync best-effort `Drop` backstop
/// force-removes the directory if cleanup never ran — e.g. a panicking task).
pub struct WorktreeGuard {
    repo_root: PathBuf,
    path: PathBuf,
    name: String,
    defused: bool,
}

/// Run a git command, capturing stdout; non-zero exit becomes `Error::Agent`
/// carrying stderr (the caller's context names the operation).
async fn git(cwd: &Path, args: &[&str]) -> Result<String, Error> {
    let out = tokio::process::Command::new("git")
        .current_dir(cwd)
        .args(args)
        .output()
        .await
        .map_err(|e| Error::Agent(format!("git not runnable: {e}")))?;
    if !out.status.success() {
        return Err(Error::Agent(format!(
            "git {} failed: {}",
            args.first().unwrap_or(&"?"),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Keep worktree/branch names shell- and ref-safe.
pub(crate) fn sanitize_name(label: &str) -> String {
    let mut s: String = label
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    s.truncate(40);
    let s = s.trim_matches('-').to_string();
    if s.is_empty() { "agent".into() } else { s }
}

impl WorktreeGuard {
    /// Create a detached worktree of `repo_root`'s HEAD under the system temp
    /// dir. Deterministic path per `name`; a stale leftover (crashed run) is
    /// force-removed first — collision-tolerant, never randomized.
    pub(crate) async fn create(repo_root: &Path, name: &str) -> Result<Self, Error> {
        let base = std::env::temp_dir().join("heartbit-worktrees");
        std::fs::create_dir_all(&base)
            .map_err(|e| Error::Agent(format!("worktree base dir: {e}")))?;
        let path = base.join(name);
        if path.exists() {
            // Stale from a previous crash: detach it from git's bookkeeping
            // and clear the directory, then recreate fresh.
            let p = path.to_string_lossy().to_string();
            let _ = git(repo_root, &["worktree", "remove", "--force", &p]).await;
            let _ = std::fs::remove_dir_all(&path);
            let _ = git(repo_root, &["worktree", "prune"]).await;
        }
        let p = path.to_string_lossy().to_string();
        git(repo_root, &["worktree", "add", "--detach", &p])
            .await
            .map_err(|e| Error::Agent(format!("worktree create '{name}': {e}")))?;
        Ok(Self {
            repo_root: repo_root.to_path_buf(),
            path,
            name: name.to_string(),
            defused: false,
        })
    }

    /// The isolated working directory the leaf should run in.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Finish the worktree: prune if clean; commit dirty state onto
    /// `hb/<name>` (work preserved in the parent repo) then remove.
    pub(crate) async fn cleanup(mut self) -> Result<Disposition, Error> {
        self.defused = true;
        let status = git(
            &self.path,
            &["status", "--porcelain", "--untracked-files=all"],
        )
        .await?;
        let p = self.path.to_string_lossy().to_string();
        if status.trim().is_empty() {
            git(&self.repo_root, &["worktree", "remove", &p]).await?;
            return Ok(Disposition::Pruned);
        }
        let branch = format!("hb/{}", self.name);
        git(&self.path, &["checkout", "-b", &branch]).await?;
        git(&self.path, &["add", "-A"]).await?;
        git(
            &self.path,
            &[
                "-c",
                "user.name=heartbit",
                "-c",
                "user.email=heartbit@local",
                "commit",
                "-m",
                &format!("heartbit worktree snapshot ({})", self.name),
            ],
        )
        .await?;
        git(&self.repo_root, &["worktree", "remove", "--force", &p]).await?;
        Ok(Disposition::Kept { branch })
    }
}

impl Drop for WorktreeGuard {
    /// Best-effort backstop only — `cleanup()` is the real path. A guard
    /// dropped without cleanup (panic/abort) force-removes the directory so
    /// the deterministic name is reusable next run; dirty state is lost here,
    /// which is why combinators must call `cleanup()`.
    fn drop(&mut self) {
        if self.defused {
            return;
        }
        let p = self.path.to_string_lossy().to_string();
        let _ = std::process::Command::new("git")
            .current_dir(&self.repo_root)
            .args(["worktree", "remove", "--force", &p])
            .output();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A throwaway repo with one commit (worktrees need a HEAD).
    async fn fixture_repo() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        for args in [
            vec!["init", "-q"],
            vec!["config", "user.name", "t"],
            vec!["config", "user.email", "t@t"],
            vec!["commit", "--allow-empty", "-q", "-m", "init"],
        ] {
            git(&root, &args.iter().map(|s| &**s).collect::<Vec<_>>())
                .await
                .unwrap();
        }
        (dir, root)
    }

    #[tokio::test]
    async fn clean_worktree_is_pruned() {
        let (_keep, root) = fixture_repo().await;
        let guard = WorktreeGuard::create(&root, "t-clean-1").await.unwrap();
        let wt = guard.path().to_path_buf();
        assert!(wt.exists(), "worktree dir created");
        assert_ne!(wt, root, "isolated from the repo root");
        let disp = guard.cleanup().await.unwrap();
        assert_eq!(disp, Disposition::Pruned);
        assert!(!wt.exists(), "pruned without residue");
    }

    #[tokio::test]
    async fn dirty_worktree_is_kept_on_a_branch() {
        let (_keep, root) = fixture_repo().await;
        let guard = WorktreeGuard::create(&root, "t-dirty-1").await.unwrap();
        std::fs::write(guard.path().join("new.txt"), "work").unwrap();
        let disp = guard.cleanup().await.unwrap();
        assert_eq!(
            disp,
            Disposition::Kept {
                branch: "hb/t-dirty-1".into()
            }
        );
        // The branch in the PARENT repo carries the file.
        let show = git(&root, &["show", "hb/t-dirty-1:new.txt"]).await.unwrap();
        assert_eq!(show, "work");
    }

    #[tokio::test]
    async fn stale_leftover_is_replaced_not_fatal() {
        let (_keep, root) = fixture_repo().await;
        let g1 = WorktreeGuard::create(&root, "t-stale-1").await.unwrap();
        let path = g1.path().to_path_buf();
        std::mem::forget(g1); // simulate a crash: no cleanup, no Drop
        assert!(path.exists());
        let g2 = WorktreeGuard::create(&root, "t-stale-1").await.unwrap();
        assert_eq!(g2.path(), path, "deterministic name reused");
        g2.cleanup().await.unwrap();
    }

    #[test]
    fn sanitize_keeps_names_ref_safe() {
        assert_eq!(
            sanitize_name("research:angle: éà x"),
            "research-angle-----x"
        );
        assert_eq!(sanitize_name(""), "agent");
        assert!(sanitize_name(&"y".repeat(100)).len() <= 40);
    }
}
