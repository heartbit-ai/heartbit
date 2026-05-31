//! Canonical skill **search-directory resolver** — the single source of truth
//! shared by [`SkillTool`](crate::tool) (Level-2 load) and
//! [`SkillRegistry`](super::SkillRegistry) (Level-1 catalog).
//!
//! Both must agree on *where* skills live, otherwise the catalog advertised in
//! the system prompt could list a skill the tool cannot load (a dead link), or
//! the tool could load a skill the model was never told about. This module is
//! that agreement: [`search_dirs`] computes the ordered list of directories that
//! may each contain `<name>/SKILL.md`, and [`catalog_for`] turns that into the
//! Level-1 system-prompt section.

use std::path::{Path, PathBuf};

use super::registry::SkillRegistry;

/// Depth cap on the upward `.git`-root walk (security: do not pick up a
/// `.claude/skills` from `/home` or `/` when `cwd` is deep under them).
const MAX_WALK_DEPTH: usize = 8;

/// Resolve the ordered skill search directories. `extra` directories (e.g. from
/// config) come **first** so they take precedence; then the conventional
/// `.opencode/skills` and `.claude/skills` for `root` (or `cwd` when `None`),
/// walking up to the git root or [`MAX_WALK_DEPTH`]; finally the global
/// `~/.config/heartbit/skills`. Each returned directory is a *parent* expected
/// to contain `<skill-name>/SKILL.md` children.
pub fn search_dirs(root: Option<&Path>, extra: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = extra.to_vec();

    let cwd = root
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let mut current = cwd.as_path();
    let mut depth = 0usize;
    loop {
        dirs.push(current.join(".opencode").join("skills"));
        dirs.push(current.join(".claude").join("skills"));

        if current.join(".git").exists() {
            break;
        }
        depth += 1;
        if depth >= MAX_WALK_DEPTH {
            break;
        }
        match current.parent() {
            Some(parent) if parent != current => current = parent,
            _ => break,
        }
    }

    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(
            PathBuf::from(home)
                .join(".config")
                .join("heartbit")
                .join("skills"),
        );
    }

    dirs
}

/// Discover skills from the canonical [`search_dirs`] and render the Level-1
/// system-prompt catalog section, or `None` if no skills are found.
///
/// SECURITY: this performs the ancestor + `$HOME` walk, so it can surface a
/// `SKILL.md` from a parent directory or the user's home config. Auto-injecting
/// such descriptions into a system prompt without the operator opting in is a
/// trust-boundary / prompt-injection risk. Use this only where the caller has
/// already opted into broad discovery; for always-on entry points prefer
/// [`catalog_from_dirs`] over an EXPLICIT, operator-provided dir list.
pub fn catalog_for(root: Option<&Path>, extra: &[PathBuf]) -> Option<String> {
    let dirs = search_dirs(root, extra);
    SkillRegistry::discover(&dirs).system_prompt_section()
}

/// Render the Level-1 catalog from an EXPLICIT, operator-provided directory list
/// only — no ancestor/`$HOME` walk. This is the opt-in form used by entry points
/// that run on every invocation (the CLI run path, config `skill_dirs`), so an
/// untrusted ancestor `.claude/skills` is never auto-injected. Returns `None`
/// when `dirs` is empty or no skills are found.
pub fn catalog_from_dirs(dirs: &[PathBuf]) -> Option<String> {
    if dirs.is_empty() {
        return None;
    }
    SkillRegistry::discover(dirs).system_prompt_section()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_skill(parent: &Path, name: &str, description: &str) {
        let dir = parent.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {description}\n---\n# {name}\n\nbody\n"),
        )
        .unwrap();
    }

    #[test]
    fn search_dirs_puts_extra_first_then_conventional() {
        let extra = vec![PathBuf::from("/custom/skills")];
        let dirs = search_dirs(Some(Path::new("/tmp/proj")), &extra);
        assert_eq!(dirs[0], PathBuf::from("/custom/skills"), "extra is first");
        assert!(
            dirs.iter().any(|d| d.ends_with(".claude/skills")),
            "conventional .claude/skills present: {dirs:?}"
        );
    }

    #[test]
    fn catalog_for_lists_skills_from_extra_dir() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "alpha", "Does alpha.");
        let section = catalog_for(Some(Path::new("/nonexistent")), &[tmp.path().to_path_buf()])
            .expect("catalog");
        assert!(section.contains("alpha: Does alpha."));
        assert!(section.contains("`skill` tool"));
    }

    #[test]
    fn catalog_for_returns_none_when_no_skills() {
        // A root with no skill dirs and an empty extra → nothing to advertise.
        let tmp = tempfile::tempdir().unwrap();
        let section = catalog_for(Some(tmp.path()), &[]);
        assert!(section.is_none());
    }

    #[test]
    fn catalog_from_dirs_is_opt_in_and_explicit() {
        // Empty list → None (no auto-discovery, the security property).
        assert!(catalog_from_dirs(&[]).is_none());

        // Explicit dir with a skill → advertised; NO ancestor/$HOME walk.
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "alpha", "Does alpha.");
        let section = catalog_from_dirs(&[tmp.path().to_path_buf()]).expect("catalog");
        assert!(section.contains("alpha: Does alpha."));
    }

    #[test]
    fn catalog_from_dirs_does_not_walk_ancestors() {
        // A skill in a PARENT dir must NOT be discovered by catalog_from_dirs
        // when only the (empty) CHILD dir is passed — proving no upward walk.
        let parent = tempfile::tempdir().unwrap();
        write_skill(parent.path(), "ancestor-skill", "Injected from parent.");
        let child = parent.path().join("child");
        std::fs::create_dir_all(&child).unwrap();
        // Pass only the child dir (which contains no skills).
        let section = catalog_from_dirs(&[child]);
        assert!(
            section.is_none(),
            "catalog_from_dirs must not climb to the parent's skills"
        );
    }

    /// The load/advertise agreement (advisor #2): a skill placed in an `extra`
    /// dir is BOTH advertised by `catalog_for` AND discoverable from the SAME
    /// `search_dirs` — so the catalog never lists a skill the tool can't load.
    #[test]
    fn catalog_and_search_dirs_agree() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "shared", "Shared skill.");
        let extra = vec![tmp.path().to_path_buf()];

        // Advertised at Level 1.
        let section = catalog_for(Some(Path::new("/nope")), &extra).expect("catalog");
        assert!(section.contains("shared: Shared skill."));

        // Loadable from the same dirs the tool would search.
        let dirs = search_dirs(Some(Path::new("/nope")), &extra);
        let reg = SkillRegistry::discover(&dirs);
        assert!(
            reg.get("shared").is_some(),
            "the advertised skill must be loadable from the same search_dirs"
        );
    }
}
