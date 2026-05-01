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
