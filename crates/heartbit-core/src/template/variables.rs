//! Prompt variable substitution — replaces `{variable}` placeholders in system prompts.

use std::collections::HashMap;

/// Substitute `{var_name}` placeholders in text.
///
/// Single-pass, non-recursive. Unknown variables are left as-is
/// (allows literal `{braces}` in prompts).
pub fn substitute(text: &str, variables: &HashMap<String, String>) -> String {
    if variables.is_empty() || !text.contains('{') {
        return text.to_string();
    }

    let mut result = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();

    while let Some((i, ch)) = chars.next() {
        if ch == '{' {
            // Look for closing brace
            let rest = &text[i + 1..];
            if let Some(close) = rest.find('}') {
                let var_name = &rest[..close];
                // Variable names: alphanumeric + underscores only
                if !var_name.is_empty()
                    && var_name.chars().all(|c| c.is_alphanumeric() || c == '_')
                    && let Some(value) = variables.get(var_name)
                {
                    result.push_str(value);
                    // Skip past the closing brace
                    for _ in 0..close + 1 {
                        chars.next();
                    }
                    continue;
                }
            }
            result.push(ch);
        } else {
            result.push(ch);
        }
    }

    result
}

/// Build the full variable context from built-in + custom variables.
pub fn build_variables(
    agent_name: &str,
    agent_description: &str,
    workspace: Option<&str>,
    custom: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut vars = HashMap::with_capacity(custom.len() + 5);
    vars.extend(custom.iter().map(|(k, v)| (k.clone(), v.clone())));

    // Built-ins override custom (intentional — prevents spoofing)
    vars.insert("agent_name".into(), agent_name.into());
    vars.insert("agent_description".into(), agent_description.into());
    vars.insert(
        "workspace".into(),
        workspace.unwrap_or("not configured").into(),
    );
    vars.insert(
        "date".into(),
        chrono::Utc::now().format("%Y-%m-%d").to_string(),
    );
    vars.insert(
        "datetime".into(),
        chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
    );

    vars
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitute_known_variable() {
        let mut vars = HashMap::new();
        vars.insert("name".into(), "Alice".into());
        assert_eq!(substitute("Hello {name}!", &vars), "Hello Alice!");
    }

    #[test]
    fn substitute_unknown_variable_preserved() {
        let vars = HashMap::new();
        assert_eq!(substitute("Hello {name}!", &vars), "Hello {name}!");
    }

    #[test]
    fn substitute_multiple_variables() {
        let mut vars = HashMap::new();
        vars.insert("a".into(), "1".into());
        vars.insert("b".into(), "2".into());
        assert_eq!(substitute("{a} + {b} = 3", &vars), "1 + 2 = 3");
    }

    #[test]
    fn substitute_empty_map_noop() {
        let vars = HashMap::new();
        assert_eq!(substitute("no vars here", &vars), "no vars here");
    }

    #[test]
    fn substitute_no_braces_noop() {
        let mut vars = HashMap::new();
        vars.insert("x".into(), "y".into());
        assert_eq!(substitute("no braces", &vars), "no braces");
    }

    #[test]
    fn substitute_literal_braces_preserved() {
        let vars = HashMap::new();
        // JSON-like braces without valid var names
        assert_eq!(
            substitute("{\"key\": \"value\"}", &vars),
            "{\"key\": \"value\"}"
        );
    }

    #[test]
    fn substitute_empty_braces_preserved() {
        let vars = HashMap::new();
        assert_eq!(substitute("empty {} here", &vars), "empty {} here");
    }

    #[test]
    fn substitute_nested_braces_preserved() {
        let mut vars = HashMap::new();
        vars.insert("x".into(), "val".into());
        // Only the inner valid var is replaced (outer brace is literal)
        assert_eq!(substitute("{{x}}", &vars), "{val}");
    }

    #[test]
    fn substitute_underscore_in_var_name() {
        let mut vars = HashMap::new();
        vars.insert("agent_name".into(), "coder".into());
        assert_eq!(substitute("I am {agent_name}", &vars), "I am coder");
    }

    #[test]
    fn build_variables_includes_builtins() {
        let vars = build_variables(
            "test-agent",
            "A test agent",
            Some("/workspace"),
            &HashMap::new(),
        );
        assert_eq!(vars.get("agent_name").unwrap(), "test-agent");
        assert_eq!(vars.get("agent_description").unwrap(), "A test agent");
        assert_eq!(vars.get("workspace").unwrap(), "/workspace");
        assert!(vars.contains_key("date"));
        assert!(vars.contains_key("datetime"));
    }

    #[test]
    fn build_variables_custom_merged() {
        let mut custom = HashMap::new();
        custom.insert("project".into(), "heartbit".into());
        let vars = build_variables("a", "d", None, &custom);
        assert_eq!(vars.get("project").unwrap(), "heartbit");
        assert_eq!(vars.get("workspace").unwrap(), "not configured");
    }

    #[test]
    fn build_variables_builtins_override_custom() {
        let mut custom = HashMap::new();
        custom.insert("agent_name".into(), "spoofed".into());
        let vars = build_variables("real", "d", None, &custom);
        assert_eq!(vars.get("agent_name").unwrap(), "real");
    }
}
