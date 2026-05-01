use crate::error::Error;

use super::AgentTemplate;

/// Embedded template data (compile-time).
static TEMPLATES: &[(&str, &str)] = &[
    ("coder", include_str!("../../templates/coder.toml")),
    (
        "researcher",
        include_str!("../../templates/researcher.toml"),
    ),
    ("planner", include_str!("../../templates/planner.toml")),
    ("reviewer", include_str!("../../templates/reviewer.toml")),
    ("debugger", include_str!("../../templates/debugger.toml")),
    ("writer", include_str!("../../templates/writer.toml")),
    ("ops", include_str!("../../templates/ops.toml")),
    (
        "orchestrator",
        include_str!("../../templates/orchestrator.toml"),
    ),
    (
        "security-auditor",
        include_str!("../../templates/security-auditor.toml"),
    ),
    (
        "test-engineer",
        include_str!("../../templates/test-engineer.toml"),
    ),
    ("architect", include_str!("../../templates/architect.toml")),
    (
        "data-scientist",
        include_str!("../../templates/data-scientist.toml"),
    ),
    ("analyst", include_str!("../../templates/analyst.toml")),
    (
        "customer-support",
        include_str!("../../templates/customer-support.toml"),
    ),
    (
        "translator",
        include_str!("../../templates/translator.toml"),
    ),
];

/// Resolve a template name to its parsed `AgentTemplate`.
///
/// Discovery order: bundled → `~/.config/heartbit/templates/{name}.toml`
/// → `.heartbit/templates/{name}.toml` (walk up to git root).
pub fn resolve_template(name: &str) -> Result<AgentTemplate, Error> {
    // Prevent path traversal
    if name.contains('/') || name.contains('\\') || name.contains("..") || name.is_empty() {
        return Err(Error::Config(format!(
            "invalid template name '{name}': must not contain path separators or '..'"
        )));
    }

    // 1. Check bundled templates
    for (key, toml_str) in TEMPLATES {
        if *key == name {
            return parse_template(name, toml_str);
        }
    }

    // 2. Check filesystem
    let search_dirs = collect_template_search_dirs();
    for dir in &search_dirs {
        let path = dir.join(format!("{name}.toml"));
        if path.is_file() {
            let content = std::fs::read_to_string(&path)
                .map_err(|e| Error::Config(format!("failed to read template '{name}': {e}")))?;
            return parse_template(name, &content);
        }
    }

    Err(Error::Config(format!(
        "unknown template '{name}'. Available templates: {}",
        known_templates().join(", ")
    )))
}

/// Returns the list of all known template names.
pub fn known_templates() -> Vec<&'static str> {
    TEMPLATES.iter().map(|(k, _)| *k).collect()
}

/// Parse a template TOML string into an `AgentTemplate`.
fn parse_template(name: &str, toml_str: &str) -> Result<AgentTemplate, Error> {
    toml::from_str(toml_str)
        .map_err(|e| Error::Config(format!("failed to parse template '{name}': {e}")))
}

/// Collect filesystem directories to search for templates.
fn collect_template_search_dirs() -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();

    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let mut current = cwd.as_path();
    loop {
        dirs.push(current.join(".heartbit").join("templates"));

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
            std::path::PathBuf::from(home)
                .join(".config")
                .join("heartbit")
                .join("templates"),
        );
    }

    dirs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_coder_template() {
        let template = resolve_template("coder").expect("should resolve coder");
        assert!(
            template
                .meta
                .description
                .starts_with("Expert software engineer")
        );
        assert!(
            !template
                .agent
                .system_prompt
                .as_deref()
                .unwrap_or("")
                .is_empty()
        );
    }

    #[test]
    fn resolve_all_templates() {
        for name in known_templates() {
            let template = resolve_template(name)
                .unwrap_or_else(|e| panic!("failed to resolve template '{name}': {e}"));
            assert!(
                !template.meta.description.is_empty(),
                "template '{name}' has empty description"
            );
        }
    }

    #[test]
    fn resolve_unknown_template_returns_error() {
        let err = resolve_template("nonexistent").unwrap_err();
        assert!(err.to_string().contains("unknown template"));
        assert!(err.to_string().contains("coder")); // lists available
    }

    #[test]
    fn resolve_rejects_path_traversal() {
        assert!(resolve_template("../etc").is_err());
        assert!(resolve_template("foo/bar").is_err());
        assert!(resolve_template("").is_err());
    }

    #[test]
    fn known_templates_returns_all() {
        let templates = known_templates();
        assert_eq!(templates.len(), 15);
        assert!(templates.contains(&"coder"));
        assert!(templates.contains(&"translator"));
    }
}
