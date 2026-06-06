//! Project-context (agent instruction) loading. Walks up from the working
//! directory to the git/filesystem root collecting one context file per
//! directory, then the global config. Supports the cross-tool standards.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Project-context filenames, in priority order: the cross-tool `AGENTS.md`
/// standard first, then Claude Code's `CLAUDE.md`, then heartbit's native
/// `HEARTBIT.md`. Within a directory, the highest-priority present file is used.
pub const CONTEXT_FILENAMES: &[&str] = &["AGENTS.md", "CLAUDE.md", "HEARTBIT.md"];

/// Pick the highest-priority context file present in `dir` (or `None`).
fn context_file_in(dir: &Path) -> Option<PathBuf> {
    CONTEXT_FILENAMES
        .iter()
        .map(|name| dir.join(name))
        .find(|p| p.is_file())
}

/// Discover project-context files ([`CONTEXT_FILENAMES`]) by walking up from
/// `working_dir` to the filesystem root (or git repository root), then the
/// global config directory (`~/.config/heartbit/`).
///
/// Returns paths in child-to-parent order (most specific first), then global;
/// at most one file per directory (highest priority).
pub fn discover_instruction_files(working_dir: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // Walk up from working_dir to root (or git boundary).
    let mut dir = working_dir.to_path_buf();
    loop {
        if let Some(found) = context_file_in(&dir) {
            paths.push(found);
        }
        // Stop at git root (directory containing .git).
        if dir.join(".git").exists() {
            break;
        }
        if !dir.pop() {
            break;
        }
    }

    // Global config: ~/.config/heartbit/{AGENTS,CLAUDE,HEARTBIT}.md
    if let Some(home) = home_dir()
        && let Some(found) = context_file_in(&home.join(".config").join("heartbit"))
    {
        paths.push(found);
    }

    paths
}

/// Load instruction files and concatenate their contents, deduplicating
/// by content to avoid injecting the same instructions twice (e.g., when
/// a project and its parent share symlinked files).
///
/// Returns the combined instruction text, or an empty string if no files
/// are found or all are empty.
pub fn load_instructions(paths: &[PathBuf]) -> std::io::Result<String> {
    let mut seen = HashSet::new();
    let mut sections = Vec::new();

    for path in paths {
        let content = std::fs::read_to_string(path)?;
        let content = content.trim().to_string();
        if content.is_empty() {
            continue;
        }
        // Deduplicate by content
        if seen.contains(&content) {
            continue;
        }
        seen.insert(content.clone());
        sections.push(content);
    }

    if sections.is_empty() {
        return Ok(String::new());
    }

    Ok(sections.join("\n\n---\n\n"))
}

/// Prepend instruction text to a system prompt.
///
/// If `instructions` is empty, returns the original prompt unchanged.
pub fn prepend_instructions(system_prompt: &str, instructions: &str) -> String {
    if instructions.is_empty() {
        return system_prompt.to_string();
    }
    format!("# Project Instructions\n\n{instructions}\n\n---\n\n{system_prompt}")
}

/// Get the user's home directory.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn discover_finds_heartbit_md_in_working_dir() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("HEARTBIT.md"), "instructions").unwrap();

        let paths = discover_instruction_files(dir.path());
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], dir.path().join("HEARTBIT.md"));
    }

    #[test]
    fn discover_walks_up_to_git_root() {
        let root = TempDir::new().unwrap();
        // Create a git root marker
        std::fs::create_dir(root.path().join(".git")).unwrap();
        std::fs::write(root.path().join("HEARTBIT.md"), "root instructions").unwrap();

        // Create a subdirectory with its own HEARTBIT.md
        let sub = root.path().join("src").join("module");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("HEARTBIT.md"), "module instructions").unwrap();

        let paths = discover_instruction_files(&sub);
        assert_eq!(paths.len(), 2);
        // Child first, then parent
        assert_eq!(paths[0], sub.join("HEARTBIT.md"));
        assert_eq!(paths[1], root.path().join("HEARTBIT.md"));
    }

    #[test]
    fn discover_stops_at_git_root() {
        let root = TempDir::new().unwrap();
        let project = root.path().join("project");
        std::fs::create_dir(&project).unwrap();
        std::fs::create_dir(project.join(".git")).unwrap();

        // HEARTBIT.md above the git root should NOT be found
        std::fs::write(root.path().join("HEARTBIT.md"), "above git root").unwrap();
        std::fs::write(project.join("HEARTBIT.md"), "in project").unwrap();

        let paths = discover_instruction_files(&project);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], project.join("HEARTBIT.md"));
    }

    #[test]
    fn discover_returns_empty_when_no_files() {
        let dir = TempDir::new().unwrap();
        // Create .git to stop traversal
        std::fs::create_dir(dir.path().join(".git")).unwrap();

        let paths = discover_instruction_files(dir.path());
        assert!(paths.is_empty());
    }

    #[test]
    fn discover_prefers_agents_md_over_claude_and_heartbit() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "agents").unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "claude").unwrap();
        std::fs::write(dir.path().join("HEARTBIT.md"), "heartbit").unwrap();
        let paths = discover_instruction_files(dir.path());
        assert_eq!(paths.len(), 1, "at most one file per directory");
        assert_eq!(paths[0], dir.path().join("AGENTS.md"));
    }

    #[test]
    fn discover_falls_back_to_claude_then_heartbit() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "claude").unwrap();
        std::fs::write(dir.path().join("HEARTBIT.md"), "heartbit").unwrap();
        assert_eq!(
            discover_instruction_files(dir.path()),
            vec![dir.path().join("CLAUDE.md")],
            "CLAUDE.md wins when no AGENTS.md"
        );
    }

    #[test]
    fn load_concatenates_with_separator() {
        let dir = TempDir::new().unwrap();
        let file1 = dir.path().join("a.md");
        let file2 = dir.path().join("b.md");
        std::fs::write(&file1, "First").unwrap();
        std::fs::write(&file2, "Second").unwrap();

        let result = load_instructions(&[file1, file2]).unwrap();
        assert_eq!(result, "First\n\n---\n\nSecond");
    }

    #[test]
    fn load_deduplicates_by_content() {
        let dir = TempDir::new().unwrap();
        let file1 = dir.path().join("a.md");
        let file2 = dir.path().join("b.md");
        std::fs::write(&file1, "Same content").unwrap();
        std::fs::write(&file2, "Same content").unwrap();

        let result = load_instructions(&[file1, file2]).unwrap();
        assert_eq!(result, "Same content");
    }

    #[test]
    fn load_skips_empty_files() {
        let dir = TempDir::new().unwrap();
        let file1 = dir.path().join("a.md");
        let file2 = dir.path().join("b.md");
        std::fs::write(&file1, "").unwrap();
        std::fs::write(&file2, "Content").unwrap();

        let result = load_instructions(&[file1, file2]).unwrap();
        assert_eq!(result, "Content");
    }

    #[test]
    fn load_trims_whitespace() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("a.md");
        std::fs::write(&file, "\n  Content with whitespace  \n\n").unwrap();

        let result = load_instructions(&[file]).unwrap();
        assert_eq!(result, "Content with whitespace");
    }

    #[test]
    fn load_empty_paths_returns_empty_string() {
        let result = load_instructions(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn prepend_adds_header_and_separator() {
        let result = prepend_instructions("You are an agent.", "Be safe.");
        assert_eq!(
            result,
            "# Project Instructions\n\nBe safe.\n\n---\n\nYou are an agent."
        );
    }

    #[test]
    fn prepend_noop_when_empty() {
        let result = prepend_instructions("You are an agent.", "");
        assert_eq!(result, "You are an agent.");
    }
}
