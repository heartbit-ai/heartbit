//! [`SkillRegistry`] — discover `SKILL.md` skills across directories and build a
//! Level-1 catalog for progressive disclosure.
//!
//! Discovery walks each search directory's immediate children for a
//! `<child>/SKILL.md`, parses + validates it ([`SkillManifest`]), and registers
//! it by name. Earlier directories win on a name clash (the caller orders search
//! dirs project-first, personal-last), so a project skill shadows a personal one
//! of the same name. A skill that fails to parse is skipped (it must not break
//! discovery of the others); the failure is recorded in [`SkillRegistry::errors`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::manifest::SkillManifest;

/// A discovered skill: its validated manifest and the directory it lives in.
#[derive(Debug, Clone)]
pub struct DiscoveredSkill {
    /// The parsed, validated `SKILL.md` manifest.
    pub manifest: SkillManifest,
    /// The skill's root directory (contains `SKILL.md` + any bundled files).
    pub dir: PathBuf,
}

/// A registry of discovered skills, keyed by name, plus any discovery errors.
#[derive(Debug, Default, Clone)]
pub struct SkillRegistry {
    skills: BTreeMap<String, DiscoveredSkill>,
    /// Non-fatal discovery errors (`(dir, message)`), e.g. a malformed SKILL.md.
    pub errors: Vec<(PathBuf, String)>,
}

impl SkillRegistry {
    /// Build an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Discover skills across `search_dirs` (highest precedence first). Each
    /// directory's immediate subdirectories are scanned for a `SKILL.md`. A name
    /// already registered by an earlier (higher-precedence) directory is not
    /// overwritten. Malformed skills are skipped and recorded in
    /// [`errors`](Self::errors).
    pub fn discover<I, P>(search_dirs: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut registry = SkillRegistry::new();
        for dir in search_dirs {
            registry.discover_dir(dir.as_ref());
        }
        registry
    }

    /// Scan one directory's immediate children for `<child>/SKILL.md`.
    fn discover_dir(&mut self, dir: &Path) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return, // a missing/unreadable search dir is not an error
        };
        // Sort for deterministic discovery order within a directory.
        let mut child_dirs: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        child_dirs.sort();

        for child in child_dirs {
            let skill_file = child.join("SKILL.md");
            if !skill_file.is_file() {
                continue;
            }
            let dir_name = match child.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            // Skip a name already claimed by a higher-precedence directory.
            if self.skills.contains_key(&dir_name) {
                continue;
            }
            match std::fs::read_to_string(&skill_file) {
                Ok(content) => match SkillManifest::parse(&content, &dir_name) {
                    Ok(manifest) => {
                        self.skills.insert(
                            manifest.name.clone(),
                            DiscoveredSkill {
                                manifest,
                                dir: child.clone(),
                            },
                        );
                    }
                    Err(e) => self.errors.push((child.clone(), e.to_string())),
                },
                Err(e) => self.errors.push((child.clone(), e.to_string())),
            }
        }
    }

    /// Number of discovered skills.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Whether the registry has no skills.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Look up a discovered skill by name.
    pub fn get(&self, name: &str) -> Option<&DiscoveredSkill> {
        self.skills.get(name)
    }

    /// Discovered skills, in name order.
    pub fn skills(&self) -> impl Iterator<Item = &DiscoveredSkill> {
        self.skills.values()
    }

    /// Render the Level-1 catalog: one `name: description` line per skill, in
    /// name order. Returns an empty string when no skills are registered (so the
    /// caller can inject nothing rather than an empty header).
    pub fn catalog(&self) -> String {
        self.skills
            .values()
            .map(|s| s.manifest.catalog_line())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Render the catalog wrapped in a system-prompt section, or `None` if empty.
    /// The section instructs the model to invoke the `skill` tool by name to load
    /// a skill's full instructions on demand (progressive disclosure level 2).
    pub fn system_prompt_section(&self) -> Option<String> {
        if self.skills.is_empty() {
            return None;
        }
        Some(format!(
            "## Available skills\n\nThe following skills are available. When a task \
             matches a skill's description, invoke the `skill` tool with its name to \
             load its full instructions before proceeding:\n\n{}",
            self.catalog()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create `root/<name>/SKILL.md` with the given content.
    fn write_skill(root: &Path, name: &str, content: &str) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), content).unwrap();
    }

    fn skill_md(name: &str, description: &str) -> String {
        format!("---\nname: {name}\ndescription: {description}\n---\n# {name}\n\nbody\n")
    }

    #[test]
    fn discovers_valid_skills() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(
            tmp.path(),
            "alpha",
            &skill_md("alpha", "Does alpha things."),
        );
        write_skill(tmp.path(), "beta", &skill_md("beta", "Does beta things."));

        let reg = SkillRegistry::discover([tmp.path()]);
        assert_eq!(reg.len(), 2);
        assert!(reg.get("alpha").is_some());
        assert!(reg.get("beta").is_some());
        assert!(reg.errors.is_empty());
    }

    #[test]
    fn catalog_is_name_colon_description_sorted() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "zeta", &skill_md("zeta", "Z."));
        write_skill(tmp.path(), "alpha", &skill_md("alpha", "A."));
        let reg = SkillRegistry::discover([tmp.path()]);
        assert_eq!(reg.catalog(), "alpha: A.\nzeta: Z.");
    }

    #[test]
    fn project_dir_shadows_personal_on_name_clash() {
        let project = tempfile::tempdir().unwrap();
        let personal = tempfile::tempdir().unwrap();
        write_skill(project.path(), "dup", &skill_md("dup", "PROJECT version."));
        write_skill(
            personal.path(),
            "dup",
            &skill_md("dup", "PERSONAL version."),
        );

        // Project dir first => higher precedence => wins.
        let reg = SkillRegistry::discover([project.path(), personal.path()]);
        assert_eq!(reg.len(), 1);
        assert_eq!(
            reg.get("dup").unwrap().manifest.description,
            "PROJECT version."
        );
    }

    #[test]
    fn malformed_skill_is_skipped_not_fatal() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "good", &skill_md("good", "Fine."));
        // name != dir_name => invalid manifest.
        write_skill(tmp.path(), "bad", &skill_md("wrong-name", "Broken."));

        let reg = SkillRegistry::discover([tmp.path()]);
        assert_eq!(reg.len(), 1, "the good skill is still discovered");
        assert!(reg.get("good").is_some());
        assert_eq!(reg.errors.len(), 1, "the bad skill is recorded as an error");
        assert!(reg.errors[0].1.contains("directory name"));
    }

    #[test]
    fn missing_search_dir_is_ignored() {
        let reg = SkillRegistry::discover([Path::new("/nonexistent/skills/dir")]);
        assert!(reg.is_empty());
        assert!(reg.errors.is_empty());
    }

    #[test]
    fn dir_without_skill_md_is_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("not-a-skill")).unwrap();
        std::fs::write(tmp.path().join("not-a-skill").join("README.md"), "hi").unwrap();
        let reg = SkillRegistry::discover([tmp.path()]);
        assert!(reg.is_empty());
    }

    #[test]
    fn empty_registry_has_no_system_prompt_section() {
        let reg = SkillRegistry::new();
        assert!(reg.system_prompt_section().is_none());
        assert_eq!(reg.catalog(), "");
    }

    #[test]
    fn system_prompt_section_lists_skills_and_mentions_skill_tool() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "alpha", &skill_md("alpha", "Does alpha."));
        let reg = SkillRegistry::discover([tmp.path()]);
        let section = reg.system_prompt_section().expect("non-empty");
        assert!(section.contains("alpha: Does alpha."));
        assert!(section.contains("`skill` tool"));
    }
}
