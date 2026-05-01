use std::path::PathBuf;

use crate::error::Error;
use crate::tool::builtins::floor_char_boundary;

/// Parsed skill content from a SKILL.md file.
#[derive(Debug, Clone)]
pub struct SkillContent {
    pub name: String,
    pub description: Option<String>,
    pub content: String,
    pub max_inject_tokens: Option<usize>,
}

/// Embedded skill data (compile-time).
static BUNDLED_SKILLS: &[(&str, &str)] = &[
    (
        "rust-expert",
        include_str!("../../skills/rust-expert/SKILL.md"),
    ),
    (
        "python-expert",
        include_str!("../../skills/python-expert/SKILL.md"),
    ),
    (
        "typescript-expert",
        include_str!("../../skills/typescript-expert/SKILL.md"),
    ),
    ("docker", include_str!("../../skills/docker/SKILL.md")),
    (
        "kubernetes",
        include_str!("../../skills/kubernetes/SKILL.md"),
    ),
    ("security", include_str!("../../skills/security/SKILL.md")),
    (
        "sql-expert",
        include_str!("../../skills/sql-expert/SKILL.md"),
    ),
    (
        "api-design",
        include_str!("../../skills/api-design/SKILL.md"),
    ),
    ("testing", include_str!("../../skills/testing/SKILL.md")),
    (
        "git-expert",
        include_str!("../../skills/git-expert/SKILL.md"),
    ),
];

/// Load a single skill by name.
///
/// Discovery order: bundled → `~/.config/heartbit/skills/{name}/SKILL.md`
/// → `.heartbit/templates/{name}.toml` (walk up to git root).
pub fn load_skill(name: &str) -> Result<SkillContent, Error> {
    // Prevent path traversal
    if name.contains('/') || name.contains('\\') || name.contains("..") || name.is_empty() {
        return Err(Error::Config(format!(
            "invalid skill name '{name}': must not contain path separators or '..'"
        )));
    }

    // 1. Check bundled skills
    for (key, md) in BUNDLED_SKILLS {
        if *key == name {
            return parse_skill_md(name, md);
        }
    }

    // 2. Check filesystem
    let search_dirs = collect_skill_search_dirs();
    for dir in &search_dirs {
        let skill_file = dir.join(name).join("SKILL.md");
        if skill_file.is_file() {
            let content = std::fs::read_to_string(&skill_file)
                .map_err(|e| Error::Config(format!("failed to read skill '{name}': {e}")))?;
            return parse_skill_md(name, &content);
        }
    }

    Err(Error::Config(format!(
        "unknown skill '{name}'. Available skills: {}",
        known_skills().join(", ")
    )))
}

/// Load multiple skills and format them as a single injection block.
///
/// Returns empty string if `names` is empty.
pub fn load_skills(names: &[String]) -> Result<String, Error> {
    if names.is_empty() {
        return Ok(String::new());
    }

    let mut sections = Vec::with_capacity(names.len());
    for name in names {
        let skill = load_skill(name)?;
        let mut section = format!("### {name}\n\n{}", skill.content);

        // Truncate if max_inject_tokens is set (rough: 1 token ≈ 4 chars)
        if let Some(max_tokens) = skill.max_inject_tokens {
            let max_chars = max_tokens * 4;
            if section.len() > max_chars {
                section.truncate(floor_char_boundary(&section, max_chars));
                section.push_str("\n[truncated]");
            }
        }

        sections.push(section);
    }

    Ok(format!("\n\n## Loaded Skills\n\n{}", sections.join("\n\n")))
}

/// Returns the list of all known skill names (bundled + filesystem).
pub fn known_skills() -> Vec<&'static str> {
    // For simplicity, return only bundled skill names.
    // Filesystem skills are discovered dynamically.
    BUNDLED_SKILLS.iter().map(|(k, _)| *k).collect()
}

/// Parse a SKILL.md file with optional TOML frontmatter.
fn parse_skill_md(name: &str, raw: &str) -> Result<SkillContent, Error> {
    let trimmed = raw.trim();

    if let Some(rest) = trimmed.strip_prefix("---") {
        // Has frontmatter
        if let Some(end) = rest.find("\n---") {
            let frontmatter = &rest[..end];
            let body = rest[end + 4..].trim();

            // Parse frontmatter as TOML
            let meta: SkillFrontmatter = toml::from_str(frontmatter).map_err(|e| {
                Error::Config(format!(
                    "failed to parse frontmatter for skill '{name}': {e}"
                ))
            })?;

            return Ok(SkillContent {
                name: meta.name.unwrap_or_else(|| name.to_string()),
                description: meta.description,
                content: body.to_string(),
                max_inject_tokens: meta.max_inject_tokens,
            });
        }
    }

    // No frontmatter — bare Markdown
    Ok(SkillContent {
        name: name.to_string(),
        description: None,
        content: trimmed.to_string(),
        max_inject_tokens: None,
    })
}

#[derive(serde::Deserialize)]
struct SkillFrontmatter {
    name: Option<String>,
    description: Option<String>,
    #[serde(default)]
    max_inject_tokens: Option<usize>,
    // tags are parsed but not used at injection time
    #[allow(dead_code)]
    #[serde(default)]
    tags: Vec<String>,
}

/// Collect filesystem directories to search for skills.
fn collect_skill_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut current = cwd.as_path();
    loop {
        dirs.push(current.join(".opencode").join("skills"));
        dirs.push(current.join(".claude").join("skills"));
        dirs.push(current.join(".heartbit").join("skills"));

        if current.join(".git").exists() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_skill_with_frontmatter() {
        let md = "---\nname = \"test\"\ndescription = \"A test skill\"\ntags = [\"a\"]\nmax_inject_tokens = 1000\n---\n\n# Content\n\nHello world";
        let skill = parse_skill_md("fallback", md).unwrap();
        assert_eq!(skill.name, "test");
        assert_eq!(skill.description.as_deref(), Some("A test skill"));
        assert_eq!(skill.max_inject_tokens, Some(1000));
        assert!(skill.content.contains("# Content"));
    }

    #[test]
    fn parse_skill_without_frontmatter() {
        let md = "# Just Markdown\n\nNo frontmatter here.";
        let skill = parse_skill_md("bare", md).unwrap();
        assert_eq!(skill.name, "bare");
        assert!(skill.description.is_none());
        assert!(skill.content.contains("Just Markdown"));
    }

    #[test]
    fn load_bundled_skill() {
        let skill = load_skill("rust-expert").unwrap();
        assert_eq!(skill.name, "rust-expert");
        assert!(!skill.content.is_empty());
    }

    #[test]
    fn load_all_bundled_skills() {
        for name in known_skills() {
            let skill =
                load_skill(name).unwrap_or_else(|e| panic!("failed to load skill '{name}': {e}"));
            assert!(
                !skill.content.is_empty(),
                "skill '{name}' has empty content"
            );
        }
    }

    #[test]
    fn load_unknown_skill_error() {
        let err = load_skill("nonexistent-skill-xyz").unwrap_err();
        assert!(err.to_string().contains("unknown skill"));
        assert!(err.to_string().contains("rust-expert")); // lists available
    }

    #[test]
    fn load_skill_rejects_path_traversal() {
        assert!(load_skill("../etc").is_err());
        assert!(load_skill("foo/bar").is_err());
        assert!(load_skill("").is_err());
    }

    #[test]
    fn load_skills_empty_names() {
        let result = load_skills(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn load_skills_formats_injection() {
        let result = load_skills(&["rust-expert".into()]).unwrap();
        assert!(result.contains("## Loaded Skills"));
        assert!(result.contains("### rust-expert"));
    }

    #[test]
    fn load_skills_unknown_returns_error() {
        let err = load_skills(&["nonexistent".into()]).unwrap_err();
        assert!(err.to_string().contains("unknown skill"));
    }

    #[test]
    fn floor_char_boundary_ascii() {
        assert_eq!(floor_char_boundary("hello", 3), 3);
        assert_eq!(floor_char_boundary("hello", 10), 5);
    }

    #[test]
    fn floor_char_boundary_utf8() {
        let s = "héllo"; // é is 2 bytes
        assert_eq!(floor_char_boundary(s, 2), 1); // before é
    }

    #[test]
    fn known_skills_returns_all() {
        let skills = known_skills();
        assert_eq!(skills.len(), 10);
        assert!(skills.contains(&"rust-expert"));
        assert!(skills.contains(&"git-expert"));
    }
}
