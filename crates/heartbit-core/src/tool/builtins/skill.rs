use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

pub struct SkillTool {
    /// Override the starting directory for skill search. When `None`, uses `cwd`.
    /// Exposed for testing without mutating the process-global cwd.
    search_root: Option<PathBuf>,
}

impl SkillTool {
    pub fn new() -> Self {
        Self { search_root: None }
    }

    #[cfg(test)]
    fn with_search_root(root: PathBuf) -> Self {
        Self {
            search_root: Some(root),
        }
    }
}

impl Tool for SkillTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "skill".into(),
            description:
                "Load a skill definition from SKILL.md files. Searches .opencode/skills/, \
                          .claude/skills/, and ~/.config/heartbit/skills/ directories."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name (matches directory name)"
                    }
                },
                "required": ["name"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let name = input
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("name is required".into()))?;

            // Prevent path traversal attacks
            if name.contains('/') || name.contains('\\') || name.contains("..") || name.is_empty() {
                return Ok(ToolOutput::error(
                    "Invalid skill name: must not contain path separators or '..'",
                ));
            }

            // Collect search directories
            let search_dirs = collect_search_dirs(self.search_root.as_deref());

            for dir in &search_dirs {
                let skill_dir = dir.join(name);
                let skill_file = skill_dir.join("SKILL.md");

                if skill_file.exists() {
                    let content = tokio::fs::read_to_string(&skill_file)
                        .await
                        .map_err(|e| Error::Agent(format!("Cannot read SKILL.md: {e}")))?;

                    // List sibling files
                    let siblings = list_siblings(&skill_dir);

                    let mut output = format!("# Skill: {name}\n\n{content}");

                    if !siblings.is_empty() {
                        output.push_str("\n\n## Sibling files:\n");
                        for s in &siblings {
                            output.push_str(&format!("- {s}\n"));
                        }
                    }

                    return Ok(ToolOutput::success(output));
                }
            }

            // Not found — list available skills
            let available = list_available_skills(&search_dirs);
            if available.is_empty() {
                Ok(ToolOutput::error(format!(
                    "Skill '{name}' not found. No skills are installed."
                )))
            } else {
                Ok(ToolOutput::error(format!(
                    "Skill '{name}' not found. Available skills: {}",
                    available.join(", ")
                )))
            }
        })
    }
}

fn collect_search_dirs(override_root: Option<&Path>) -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    // Walk up from the starting directory to git root
    let cwd = override_root
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let mut current = cwd.as_path();
    loop {
        dirs.push(current.join(".opencode").join("skills"));
        dirs.push(current.join(".claude").join("skills"));

        // Stop at git root or filesystem root
        if current.join(".git").exists() {
            break;
        }
        match current.parent() {
            Some(parent) if parent != current => current = parent,
            _ => break,
        }
    }

    // Global config directory
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

fn list_siblings(skill_dir: &Path) -> Vec<String> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(skill_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_str().unwrap_or("");
            if name_str != "SKILL.md" {
                files.push(name_str.to_string());
            }
        }
    }
    files.sort();
    files
}

fn list_available_skills(search_dirs: &[PathBuf]) -> Vec<String> {
    let mut skills = std::collections::BTreeSet::new();

    for dir in search_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    let skill_file = entry.path().join("SKILL.md");
                    if skill_file.exists()
                        && let Some(name) = entry.file_name().to_str()
                    {
                        skills.insert(name.to_string());
                    }
                }
            }
        }
    }

    skills.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tool = SkillTool::new();
        assert_eq!(tool.definition().name, "skill");
    }

    #[tokio::test]
    async fn skill_not_found() {
        let tool = SkillTool::new();
        let result = tool
            .execute(json!({"name": "nonexistent_skill_12345"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn skill_rejects_path_traversal() {
        let tool = SkillTool::new();

        // Directory traversal
        let result = tool.execute(json!({"name": "../../etc"})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Invalid skill name"));

        // Forward slash
        let result = tool.execute(json!({"name": "foo/bar"})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Invalid skill name"));

        // Backslash
        let result = tool.execute(json!({"name": "foo\\bar"})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Invalid skill name"));

        // Empty
        let result = tool.execute(json!({"name": ""})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Invalid skill name"));
    }

    #[tokio::test]
    async fn skill_loads_from_directory() {
        let dir = tempfile::tempdir().unwrap();
        let skills_dir = dir
            .path()
            .join(".opencode")
            .join("skills")
            .join("test-skill");
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(skills_dir.join("SKILL.md"), "# Test Skill\nDoes testing.").unwrap();
        std::fs::write(skills_dir.join("helper.sh"), "#!/bin/bash\n").unwrap();

        let tool = SkillTool::with_search_root(dir.path().to_path_buf());
        let result = tool.execute(json!({"name": "test-skill"})).await.unwrap();

        assert!(!result.is_error, "got error: {}", result.content);
        assert!(result.content.contains("Test Skill"));
        assert!(result.content.contains("helper.sh"));
    }
}
