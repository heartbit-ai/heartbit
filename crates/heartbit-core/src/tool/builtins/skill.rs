use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

/// Builtin tool that executes pre-written prompt scripts ("skills") from disk.
///
/// Searches upward from the current working directory for a `.heartbit/skills/`
/// directory, then reads the named skill file and returns its contents as
/// instructions for the agent. This enables reusable, version-controlled
/// prompt recipes. Skill names are validated against path traversal: names
/// containing `/`, `\`, `..`, or empty strings are rejected.
#[derive(Default)]
pub struct SkillTool {
    /// Override the starting directory for skill search. When `None`, uses `cwd`.
    /// Exposed for testing without mutating the process-global cwd.
    search_root: Option<PathBuf>,
    /// Explicit, highest-precedence skill directories (e.g. from config). These
    /// are searched before the conventional `.opencode/.claude/skills` walk, and
    /// are the SAME dirs the Level-1 catalog is built from, so the catalog never
    /// advertises a skill this tool cannot load.
    extra_dirs: Vec<PathBuf>,
}

impl SkillTool {
    /// Create a tool that also searches `extra_dirs` (highest precedence). Pass
    /// the same directories used to build the Level-1 catalog so advertise and
    /// load stay in agreement. Use [`SkillTool::default`] for conventional
    /// discovery only (no extra dirs).
    pub fn with_dirs(extra_dirs: Vec<PathBuf>) -> Self {
        Self {
            search_root: None,
            extra_dirs,
        }
    }

    #[cfg(test)]
    fn with_search_root(root: PathBuf) -> Self {
        Self {
            search_root: Some(root),
            extra_dirs: Vec::new(),
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
        _ctx: &crate::ExecutionContext,
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

            // Collect search directories from the canonical resolver shared with
            // the Level-1 catalog (so advertise == load).
            let search_dirs =
                crate::skill::search_dirs(self.search_root.as_deref(), &self.extra_dirs);

            for dir in &search_dirs {
                let skill_dir = dir.join(name);
                let skill_file = skill_dir.join("SKILL.md");

                if skill_file.exists() {
                    let content = tokio::fs::read_to_string(&skill_file)
                        .await
                        .map_err(|e| Error::Agent(format!("Cannot read SKILL.md: {e}")))?;

                    // List sibling files
                    let siblings = list_siblings(&skill_dir);

                    // Frontmatter-aware rendering (progressive disclosure level 2):
                    // when the SKILL.md has a valid Claude-format YAML frontmatter,
                    // strip it and present the clean body plus the parsed metadata
                    // (description + allowed-tools). A SKILL.md with no/invalid
                    // frontmatter falls back to the raw content (back-compat).
                    let mut output = match crate::skill::SkillManifest::parse(&content, name) {
                        Ok(manifest) => {
                            let mut o = format!("# Skill: {name}\n\n{}", manifest.description);
                            if !manifest.allowed_tools.is_empty() {
                                o.push_str(&format!(
                                    "\n\nAllowed tools while this skill is active: {}",
                                    manifest.allowed_tools.join(", ")
                                ));
                            }
                            o.push_str("\n\n");
                            o.push_str(manifest.body.trim_start());
                            o
                        }
                        Err(_) => format!("# Skill: {name}\n\n{content}"),
                    };

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
        let tool = SkillTool::default();
        assert_eq!(tool.definition().name, "skill");
    }

    #[tokio::test]
    async fn skill_not_found() {
        let tool = SkillTool::default();
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"name": "nonexistent_skill_12345"}),
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn skill_rejects_path_traversal() {
        let tool = SkillTool::default();

        // Directory traversal
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"name": "../../etc"}),
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Invalid skill name"));

        // Forward slash
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"name": "foo/bar"}),
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Invalid skill name"));

        // Backslash
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"name": "foo\\bar"}),
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Invalid skill name"));

        // Empty
        let result = tool
            .execute(&crate::ExecutionContext::default(), json!({"name": ""}))
            .await
            .unwrap();
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
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"name": "test-skill"}),
            )
            .await
            .unwrap();

        assert!(!result.is_error, "got error: {}", result.content);
        assert!(result.content.contains("Test Skill"));
        assert!(result.content.contains("helper.sh"));
    }

    #[tokio::test]
    async fn skill_with_yaml_frontmatter_is_stripped_and_metadata_surfaced() {
        let dir = tempfile::tempdir().unwrap();
        let skills_dir = dir.path().join(".claude").join("skills").join("pdf-tool");
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(
            skills_dir.join("SKILL.md"),
            "---\nname: pdf-tool\ndescription: Extract text from PDFs. Use for PDF tasks.\nallowed-tools: read, bash\n---\n# PDF Tool\n\nStep 1: open the file.\n",
        )
        .unwrap();

        let tool = SkillTool::with_search_root(dir.path().to_path_buf());
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"name": "pdf-tool"}),
            )
            .await
            .unwrap();

        assert!(!result.is_error, "got error: {}", result.content);
        // The raw frontmatter fence must NOT leak into the model's context.
        assert!(
            !result.content.contains("---\nname:"),
            "frontmatter must be stripped: {}",
            result.content
        );
        // The body and parsed metadata are surfaced.
        assert!(result.content.contains("Extract text from PDFs"));
        assert!(result.content.contains("Step 1: open the file."));
        assert!(
            result
                .content
                .contains("Allowed tools while this skill is active: read, bash")
        );
    }

    #[tokio::test]
    async fn skill_without_frontmatter_falls_back_to_raw() {
        let dir = tempfile::tempdir().unwrap();
        let skills_dir = dir.path().join(".claude").join("skills").join("plain");
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(
            skills_dir.join("SKILL.md"),
            "# Plain\n\nNo frontmatter here.",
        )
        .unwrap();

        let tool = SkillTool::with_search_root(dir.path().to_path_buf());
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"name": "plain"}),
            )
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("No frontmatter here."));
    }
}
