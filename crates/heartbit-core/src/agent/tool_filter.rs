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
    /// Read-only: investigation tools only — no edit/write/patch, no bash
    /// (side effects). The STUDY/ANSWER mode contract (request-intent
    /// router): the model cannot call what it never received.
    ReadOnly,
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

/// Tool names available in the ReadOnly profile (STUDY/ANSWER contracts):
/// reading, searching, planning artifacts and the user-dialogue channel —
/// nothing that mutates the workspace or runs commands.
const READ_ONLY_TOOLS: &[&str] = &[
    "read",
    "grep",
    "glob",
    "list",
    "todoread",
    "todowrite",
    "webfetch",
    "websearch",
    "fetch_full_output",
    "recall_context",
    "advisor",
    "set_goal",
    "set_scope",
    "run_workflow",
    "handoff",
    "memory_recall",
    "memory_store",
    "memory_update",
    "memory_forget",
    "memory_consolidate",
    "question",
    "__respond__",
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
    "image_generate",
    "tts",
    "skill",
    "todowrite",
    "todoread",
    "twitter_post",
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

/// Filter tool definitions to match a profile.
///
/// Essential tools (memory_recall, memory_store, question) are always included.
/// The profile determines which additional tools are available.
pub fn filter_tools(tools: &[ToolDefinition], profile: ToolProfile) -> Vec<ToolDefinition> {
    match profile {
        ToolProfile::Full => tools.to_vec(),
        ToolProfile::ReadOnly => tools
            .iter()
            .filter(|t| READ_ONLY_TOOLS.contains(&t.name.as_str()))
            .cloned()
            .collect(),
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
}
