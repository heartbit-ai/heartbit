//! Dynamic tool filtering based on query classification.
//!
//! Pre-classifies queries into tool profiles to reduce the number of tool
//! definitions sent to the LLM, saving input tokens on simple requests.

use crate::llm::types::ToolDefinition;

/// Pre-classified tool profiles for common query patterns.
///
/// Each profile represents a different level of tool access,
/// from minimal (conversational) to full (all tools).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolProfile {
    /// Minimal: memory tools + question only (~5 tools, ~500 tokens).
    /// For greetings, casual chat, simple Q&A.
    Conversational,
    /// Standard: builtins + memory (~14 tools, ~2000 tokens).
    /// For tasks that need file/bash/search tools but not MCP.
    Standard,
    /// Full: all tools including MCP (~29 tools, ~4500 tokens).
    /// For tasks requiring external service integration.
    Full,
}

/// Tool names that are always included regardless of profile.
/// Memory tools are essential for context continuity.
/// `__respond__` is the synthetic structured-output tool — must never be filtered.
const ESSENTIAL_TOOLS: &[&str] = &["memory_recall", "memory_store", "question", "__respond__"];

/// Tool names included in the Conversational profile (beyond essentials).
const CONVERSATIONAL_TOOLS: &[&str] = &[
    "memory_update",
    "memory_forget",
    "memory_consolidate",
    "todoread",
];

/// Built-in tool names that indicate the Standard profile.
const BUILTIN_TOOLS: &[&str] = &[
    "bash",
    "read",
    "write",
    "edit",
    "patch",
    "glob",
    "grep",
    "list",
    "webfetch",
    "websearch",
    "skill",
    "todowrite",
    "todoread",
    // Memory tools are also builtins
    "memory_recall",
    "memory_store",
    "memory_update",
    "memory_forget",
    "memory_consolidate",
    "question",
    // Synthetic tool for structured output — must never be filtered
    "__respond__",
];

/// Keywords that signal the user wants file/code/system tools (→ Standard).
const STANDARD_KEYWORDS: &[&str] = &[
    "file",
    "files",
    "read",
    "write",
    "edit",
    "patch",
    "code",
    "bash",
    "run",
    "execute",
    "command",
    "search",
    "find",
    "grep",
    "list",
    "directory",
    "folder",
    "create",
    "delete",
    "remove",
    "compile",
    "build",
    "test",
    "debug",
    "log",
    "install",
    "deploy",
    "commit",
    "git",
    "script",
    "function",
    "class",
    "module",
    "import",
    "config",
    "configuration",
    "database",
    "query",
    "api",
    "endpoint",
    "server",
    "docker",
    "container",
    "todo",
    "task",
];

/// Keywords that signal the user wants external/MCP tools (→ Full).
const FULL_KEYWORDS: &[&str] = &[
    "mcp",
    "tool",
    "integration",
    "external",
    "service",
    "webhook",
    "slack",
    "github",
    "jira",
    "notion",
    "calendar",
    "email",
    "send",
    "post",
    "publish",
    "pipeline",
    "ci",
    "cd",
];

/// Classify a query into a tool profile using keyword heuristics.
///
/// The classification is conservative: when in doubt, it escalates to a
/// higher-access profile to avoid missing needed tools.
pub fn classify_query(query: &str, tool_names: &[&str]) -> ToolProfile {
    let lower = query.to_lowercase();

    // If the query explicitly mentions a non-builtin tool name → Full
    let has_mcp_tool_mention = tool_names
        .iter()
        .any(|name| !BUILTIN_TOOLS.contains(name) && contains_word(&lower, name));
    if has_mcp_tool_mention {
        return ToolProfile::Full;
    }

    // Check for Full keywords
    let full_score: usize = FULL_KEYWORDS
        .iter()
        .filter(|kw| contains_word(&lower, kw))
        .count();
    if full_score >= 2 {
        return ToolProfile::Full;
    }

    // Check for Standard keywords
    let standard_score: usize = STANDARD_KEYWORDS
        .iter()
        .filter(|kw| contains_word(&lower, kw))
        .count();
    if standard_score >= 1 {
        return ToolProfile::Standard;
    }

    // Check query length/complexity heuristics
    // Long queries or multi-sentence queries are more likely to need tools
    let sentence_count =
        lower.matches('.').count() + lower.matches('!').count() + lower.matches('?').count();
    if query.len() > 200 || sentence_count > 2 {
        return ToolProfile::Standard;
    }

    ToolProfile::Conversational
}

/// Filter tool definitions to match a profile.
///
/// Essential tools (memory_recall, memory_store, question) are always included.
/// The profile determines which additional tools are available.
pub fn filter_tools(tools: &[ToolDefinition], profile: ToolProfile) -> Vec<ToolDefinition> {
    match profile {
        ToolProfile::Full => tools.to_vec(),
        ToolProfile::Standard => tools
            .iter()
            .filter(|t| BUILTIN_TOOLS.contains(&t.name.as_str()))
            .cloned()
            .collect(),
        ToolProfile::Conversational => tools
            .iter()
            .filter(|t| {
                ESSENTIAL_TOOLS.contains(&t.name.as_str())
                    || CONVERSATIONAL_TOOLS.contains(&t.name.as_str())
            })
            .cloned()
            .collect(),
    }
}

/// Word-boundary aware keyword matching.
///
/// Returns true if `text` contains `word` as a standalone word (not as a
/// substring of a larger word). Uses simple char-boundary checks.
fn contains_word(text: &str, word: &str) -> bool {
    let word_lower = word.to_lowercase();
    let mut start = 0;
    while let Some(pos) = text[start..].find(&word_lower) {
        let abs_pos = start + pos;
        let before_ok = abs_pos == 0 || !text.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();
        let after_pos = abs_pos + word_lower.len();
        let after_ok =
            after_pos >= text.len() || !text.as_bytes()[after_pos].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
        start = abs_pos + 1;
        if start >= text.len() {
            break;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            description: format!("Tool: {name}"),
            input_schema: json!({"type": "object"}),
        }
    }

    fn all_tool_names() -> Vec<&'static str> {
        vec![
            "bash",
            "read",
            "write",
            "edit",
            "patch",
            "glob",
            "grep",
            "list",
            "webfetch",
            "websearch",
            "skill",
            "todowrite",
            "todoread",
            "question",
            "memory_recall",
            "memory_store",
            "memory_update",
            "memory_forget",
            "memory_consolidate",
            // MCP tools
            "slack_send",
            "github_create_issue",
            "jira_search",
        ]
    }

    #[test]
    fn classify_conversational_greetings() {
        let names = all_tool_names();
        let names_ref: Vec<&str> = names.iter().copied().collect();
        assert_eq!(
            classify_query("hello", &names_ref),
            ToolProfile::Conversational
        );
        assert_eq!(
            classify_query("hi there!", &names_ref),
            ToolProfile::Conversational
        );
        assert_eq!(
            classify_query("how are you?", &names_ref),
            ToolProfile::Conversational
        );
        assert_eq!(
            classify_query("thanks", &names_ref),
            ToolProfile::Conversational
        );
        assert_eq!(
            classify_query("good morning", &names_ref),
            ToolProfile::Conversational
        );
    }

    #[test]
    fn classify_standard_file_operations() {
        let names = all_tool_names();
        let names_ref: Vec<&str> = names.iter().copied().collect();
        assert_eq!(
            classify_query("read the file", &names_ref),
            ToolProfile::Standard
        );
        assert_eq!(
            classify_query("run the command", &names_ref),
            ToolProfile::Standard
        );
        assert_eq!(
            classify_query("search for files matching *.rs", &names_ref),
            ToolProfile::Standard
        );
        assert_eq!(
            classify_query("write a script to process data", &names_ref),
            ToolProfile::Standard
        );
        assert_eq!(
            classify_query("edit the config", &names_ref),
            ToolProfile::Standard
        );
    }

    #[test]
    fn classify_full_mcp_tool_mention() {
        let names = all_tool_names();
        let names_ref: Vec<&str> = names.iter().copied().collect();
        // Mentioning a non-builtin tool name → Full
        assert_eq!(
            classify_query("use slack_send to notify the team", &names_ref),
            ToolProfile::Full,
        );
        assert_eq!(
            classify_query("create a github_create_issue for this bug", &names_ref),
            ToolProfile::Full,
        );
    }

    #[test]
    fn classify_full_multiple_keywords() {
        let names = all_tool_names();
        let names_ref: Vec<&str> = names.iter().copied().collect();
        // Two or more full keywords → Full
        assert_eq!(
            classify_query("send an email via the integration", &names_ref),
            ToolProfile::Full,
        );
    }

    #[test]
    fn classify_standard_long_query() {
        let names = all_tool_names();
        let names_ref: Vec<&str> = names.iter().copied().collect();
        // Long query without specific keywords → Standard (complexity heuristic)
        let long_query = "I need you to help me understand the architecture of this project and explain how the different components interact with each other. Also explain the data flow between modules and the error handling strategy used throughout the codebase.";
        assert_eq!(
            classify_query(long_query, &names_ref),
            ToolProfile::Standard
        );
    }

    #[test]
    fn filter_conversational_subset() {
        let tools: Vec<ToolDefinition> = vec![
            make_tool("bash"),
            make_tool("read"),
            make_tool("write"),
            make_tool("memory_recall"),
            make_tool("memory_store"),
            make_tool("question"),
            make_tool("memory_update"),
            make_tool("todoread"),
            make_tool("slack_send"),
        ];

        let filtered = filter_tools(&tools, ToolProfile::Conversational);
        let names: Vec<&str> = filtered.iter().map(|t| t.name.as_str()).collect();

        // Essential + conversational tools only
        assert!(names.contains(&"memory_recall"));
        assert!(names.contains(&"memory_store"));
        assert!(names.contains(&"question"));
        assert!(names.contains(&"memory_update"));
        assert!(names.contains(&"todoread"));
        // NOT included
        assert!(!names.contains(&"bash"));
        assert!(!names.contains(&"read"));
        assert!(!names.contains(&"write"));
        assert!(!names.contains(&"slack_send"));
    }

    #[test]
    fn filter_standard_excludes_mcp() {
        let tools: Vec<ToolDefinition> = vec![
            make_tool("bash"),
            make_tool("read"),
            make_tool("memory_recall"),
            make_tool("slack_send"),
            make_tool("github_create_issue"),
        ];

        let filtered = filter_tools(&tools, ToolProfile::Standard);
        let names: Vec<&str> = filtered.iter().map(|t| t.name.as_str()).collect();

        assert!(names.contains(&"bash"));
        assert!(names.contains(&"read"));
        assert!(names.contains(&"memory_recall"));
        assert!(!names.contains(&"slack_send"));
        assert!(!names.contains(&"github_create_issue"));
    }

    #[test]
    fn filter_full_includes_everything() {
        let tools: Vec<ToolDefinition> = vec![
            make_tool("bash"),
            make_tool("memory_recall"),
            make_tool("slack_send"),
        ];

        let filtered = filter_tools(&tools, ToolProfile::Full);
        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn filter_preserves_essential_tools_in_conversational() {
        let tools: Vec<ToolDefinition> = vec![
            make_tool("memory_recall"),
            make_tool("memory_store"),
            make_tool("question"),
        ];

        let filtered = filter_tools(&tools, ToolProfile::Conversational);
        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn filter_preserves_respond_tool_in_conversational() {
        let tools: Vec<ToolDefinition> = vec![
            make_tool("bash"),
            make_tool("memory_recall"),
            make_tool("question"),
            make_tool("__respond__"),
            make_tool("slack_send"),
        ];

        let filtered = filter_tools(&tools, ToolProfile::Conversational);
        let names: Vec<&str> = filtered.iter().map(|t| t.name.as_str()).collect();

        assert!(
            names.contains(&"__respond__"),
            "__respond__ must survive Conversational filter"
        );
        assert!(names.contains(&"memory_recall"));
        assert!(names.contains(&"question"));
        assert!(!names.contains(&"bash"));
        assert!(!names.contains(&"slack_send"));
    }

    #[test]
    fn filter_preserves_respond_tool_in_standard() {
        let tools: Vec<ToolDefinition> = vec![
            make_tool("bash"),
            make_tool("__respond__"),
            make_tool("slack_send"),
        ];

        let filtered = filter_tools(&tools, ToolProfile::Standard);
        let names: Vec<&str> = filtered.iter().map(|t| t.name.as_str()).collect();

        assert!(
            names.contains(&"__respond__"),
            "__respond__ must survive Standard filter"
        );
        assert!(names.contains(&"bash"));
        assert!(!names.contains(&"slack_send"));
    }

    #[test]
    fn contains_word_boundary_matching() {
        assert!(contains_word("read the file", "read"));
        assert!(contains_word("file read", "read"));
        assert!(!contains_word("already", "read"));
        assert!(contains_word("run a script", "run"));
        assert!(!contains_word("running", "run"));
        assert!(contains_word("use bash to check", "bash"));
    }

    #[test]
    fn classify_empty_query() {
        let names: Vec<&str> = vec!["bash", "read"];
        assert_eq!(classify_query("", &names), ToolProfile::Conversational);
    }

    #[test]
    fn classify_no_tool_names() {
        // Even with no tool names, classification works via keywords
        assert_eq!(classify_query("read the file", &[]), ToolProfile::Standard);
        assert_eq!(classify_query("hello", &[]), ToolProfile::Conversational);
    }
}
