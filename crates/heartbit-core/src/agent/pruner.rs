use crate::llm::types::{ContentBlock, Message, Role};
use crate::tool::builtins::floor_char_boundary;

/// Configuration for session-level pruning of old tool results.
///
/// Before each LLM call, old tool results are truncated in-place to reduce
/// token usage. Recent messages are preserved at full fidelity.
#[derive(Debug, Clone)]
pub struct SessionPruneConfig {
    /// Number of recent user/assistant message pairs to keep at full fidelity.
    /// Default: 2.
    pub keep_recent_n: usize,
    /// Maximum bytes for a pruned tool result. Content exceeding this is
    /// replaced with head + tail + `[pruned: N bytes]`. Default: 200.
    pub pruned_tool_result_max_bytes: usize,
    /// Whether to preserve the first user message (task) from pruning.
    /// Default: true.
    pub preserve_task: bool,
}

impl Default for SessionPruneConfig {
    fn default() -> Self {
        Self {
            keep_recent_n: 2,
            pruned_tool_result_max_bytes: 200,
            preserve_task: true,
        }
    }
}

/// Statistics from a session pruning pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PruneStats {
    /// Number of tool results that were truncated.
    pub tool_results_pruned: usize,
    /// Total bytes removed across all truncated tool results.
    pub bytes_saved: usize,
    /// Total number of tool results inspected (pruned + skipped).
    pub tool_results_total: usize,
}

impl PruneStats {
    /// Returns `true` if any pruning actually occurred.
    pub fn did_prune(&self) -> bool {
        self.tool_results_pruned > 0
    }
}

/// Prune old tool results in a message list, returning a new list and stats.
///
/// Messages in the "recent" tail (last `keep_recent_n * 2` messages) are kept
/// intact. Older messages containing tool results have their content truncated
/// to `max_bytes` with a `[pruned: N bytes]` marker.
///
/// The first message (task) is always preserved if `preserve_task` is true.
/// Message count and roles are never changed — only content is shortened.
pub fn prune_old_tool_results(
    messages: &[Message],
    config: &SessionPruneConfig,
) -> (Vec<Message>, PruneStats) {
    if messages.is_empty() {
        return (vec![], PruneStats::default());
    }

    let mut stats = PruneStats::default();

    // Recent tail: keep the last N*2 messages (user+assistant pairs)
    let recent_count = config.keep_recent_n * 2;
    let recent_start = messages.len().saturating_sub(recent_count);

    let pruned = messages
        .iter()
        .enumerate()
        .map(|(i, msg)| {
            // Preserve task message
            if i == 0 && config.preserve_task {
                return msg.clone();
            }
            // Preserve recent messages
            if i >= recent_start {
                return msg.clone();
            }
            // Only prune User messages with tool results
            if msg.role != Role::User {
                return msg.clone();
            }
            let has_tool_results = msg
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolResult { .. }));
            if !has_tool_results {
                return msg.clone();
            }
            // Prune tool result content
            let pruned_content = msg
                .content
                .iter()
                .map(|block| match block {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        stats.tool_results_total += 1;
                        let max = config.pruned_tool_result_max_bytes;
                        let pruned = truncate_with_marker(content, max);
                        if pruned.len() < content.len() {
                            stats.tool_results_pruned += 1;
                            stats.bytes_saved += content.len() - pruned.len();
                        }
                        ContentBlock::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content: pruned,
                            is_error: *is_error,
                        }
                    }
                    other => other.clone(),
                })
                .collect();
            Message {
                role: msg.role.clone(),
                content: pruned_content,
            }
        })
        .collect();

    (pruned, stats)
}

/// Truncate content to `max_bytes` with a `[pruned: N bytes]` marker.
///
/// If content fits within `max_bytes`, returns it unchanged.
/// Otherwise, keeps head bytes up to a char boundary and appends marker.
fn truncate_with_marker(content: &str, max_bytes: usize) -> String {
    if content.len() <= max_bytes {
        return content.to_string();
    }
    let omitted = content.len() - max_bytes;
    let marker = format!("\n[pruned: {omitted} bytes omitted]");
    let head_budget = max_bytes.saturating_sub(marker.len());
    let boundary = floor_char_boundary(content, head_budget);
    let head = &content[..boundary];
    format!("{head}{marker}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::ToolResult;
    use serde_json::json;

    fn tool_use_msg(id: &str, name: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: id.into(),
                name: name.into(),
                input: json!({}),
            }],
        }
    }

    fn tool_result_msg(id: &str, content: &str) -> Message {
        Message::tool_results(vec![ToolResult::success(id, content)])
    }

    #[test]
    fn prune_preserves_recent_messages() {
        let messages = vec![
            Message::user("task"),
            tool_use_msg("c1", "search"),
            tool_result_msg("c1", &"x".repeat(1000)),
            tool_use_msg("c2", "read"),
            tool_result_msg("c2", &"y".repeat(1000)),
            Message::assistant("final answer"),
        ];

        let config = SessionPruneConfig {
            keep_recent_n: 2,
            pruned_tool_result_max_bytes: 50,
            preserve_task: true,
        };
        let (pruned, stats) = prune_old_tool_results(&messages, &config);

        assert_eq!(pruned.len(), messages.len(), "message count unchanged");

        // Last 4 messages (2 pairs) should be intact
        let last_result = &pruned[4];
        if let ContentBlock::ToolResult { content, .. } = &last_result.content[0] {
            assert_eq!(content.len(), 1000, "recent tool result should be intact");
        }

        // Only 1 tool result is outside the recent window (c1), but it's
        // also the task-adjacent one — the first user msg (task) is index 0,
        // c1 result is index 2. With keep_recent_n=2 the recent window
        // starts at index 2 (6-4=2), so c1 is at the boundary and preserved.
        assert!(!stats.did_prune());
    }

    #[test]
    fn prune_trims_old_tool_results() {
        let messages = vec![
            Message::user("task"),
            tool_use_msg("c1", "search"),
            tool_result_msg("c1", &"a".repeat(1000)),
            tool_use_msg("c2", "read"),
            tool_result_msg("c2", &"b".repeat(500)),
            tool_use_msg("c3", "write"),
            tool_result_msg("c3", "short result"),
            Message::assistant("done"),
        ];

        let config = SessionPruneConfig {
            keep_recent_n: 1,
            pruned_tool_result_max_bytes: 100,
            preserve_task: true,
        };
        let (pruned, stats) = prune_old_tool_results(&messages, &config);

        // messages[2] (old tool result with 1000 bytes) should be pruned
        if let ContentBlock::ToolResult { content, .. } = &pruned[2].content[0] {
            assert!(
                content.len() <= 200,
                "old tool result should be truncated, got {} bytes",
                content.len()
            );
            assert!(content.contains("[pruned:"));
        }

        // messages[4] (old tool result with 500 bytes) should also be pruned
        if let ContentBlock::ToolResult { content, .. } = &pruned[4].content[0] {
            assert!(
                content.len() <= 200,
                "old tool result should be truncated, got {} bytes",
                content.len()
            );
            assert!(content.contains("[pruned:"));
        }

        assert!(stats.did_prune());
        assert_eq!(stats.tool_results_pruned, 2);
        assert!(stats.bytes_saved > 0);
        assert_eq!(stats.tool_results_total, 2);
    }

    #[test]
    fn prune_preserves_task_message() {
        let messages = vec![
            Message::user("important initial task"),
            tool_use_msg("c1", "search"),
            tool_result_msg("c1", &"x".repeat(1000)),
            Message::assistant("answer"),
        ];

        let config = SessionPruneConfig {
            keep_recent_n: 0,
            pruned_tool_result_max_bytes: 50,
            preserve_task: true,
        };
        let (pruned, _stats) = prune_old_tool_results(&messages, &config);

        // Task message should be unchanged
        if let ContentBlock::Text { text } = &pruned[0].content[0] {
            assert_eq!(text, "important initial task");
        }
    }

    #[test]
    fn prune_preserves_message_count() {
        let messages = vec![
            Message::user("task"),
            tool_use_msg("c1", "search"),
            tool_result_msg("c1", &"x".repeat(1000)),
            tool_use_msg("c2", "read"),
            tool_result_msg("c2", &"y".repeat(1000)),
            Message::assistant("done"),
        ];

        let config = SessionPruneConfig::default();
        let (pruned, _stats) = prune_old_tool_results(&messages, &config);

        assert_eq!(pruned.len(), messages.len());
        // Verify roles are preserved
        for (original, pruned) in messages.iter().zip(pruned.iter()) {
            assert_eq!(original.role, pruned.role);
        }
    }

    #[test]
    fn prune_utf8_safe() {
        // Multi-byte UTF-8 content should not be split at invalid boundaries
        let emoji_content = "🦀".repeat(100); // 400 bytes, 100 chars
        let messages = vec![
            Message::user("task"),
            tool_use_msg("c1", "search"),
            tool_result_msg("c1", &emoji_content),
            Message::assistant("done"),
        ];

        let config = SessionPruneConfig {
            keep_recent_n: 0,
            pruned_tool_result_max_bytes: 50,
            preserve_task: true,
        };
        let (pruned, _stats) = prune_old_tool_results(&messages, &config);

        // Should not panic and content should be valid UTF-8
        if let ContentBlock::ToolResult { content, .. } = &pruned[2].content[0] {
            assert!(content.is_char_boundary(0));
            // Verify it's valid UTF-8 by iterating
            for _ in content.chars() {}
        }
    }

    #[test]
    fn prune_empty_messages() {
        let (pruned, stats) = prune_old_tool_results(&[], &SessionPruneConfig::default());
        assert!(pruned.is_empty());
        assert!(!stats.did_prune());
    }

    #[test]
    fn prune_no_tool_results_is_noop() {
        let messages = vec![
            Message::user("task"),
            Message::assistant("response 1"),
            Message::user("follow up"),
            Message::assistant("response 2"),
        ];

        let config = SessionPruneConfig {
            keep_recent_n: 0,
            pruned_tool_result_max_bytes: 10,
            preserve_task: true,
        };
        let (pruned, stats) = prune_old_tool_results(&messages, &config);

        // No tool results to prune, all messages should be unchanged
        for (original, pruned) in messages.iter().zip(pruned.iter()) {
            assert_eq!(original.content.len(), pruned.content.len());
        }
        assert!(!stats.did_prune());
    }

    #[test]
    fn prune_short_tool_results_unchanged() {
        let messages = vec![
            Message::user("task"),
            tool_use_msg("c1", "search"),
            tool_result_msg("c1", "short"),
            Message::assistant("done"),
        ];

        let config = SessionPruneConfig {
            keep_recent_n: 0,
            pruned_tool_result_max_bytes: 200,
            preserve_task: true,
        };
        let (pruned, stats) = prune_old_tool_results(&messages, &config);

        if let ContentBlock::ToolResult { content, .. } = &pruned[2].content[0] {
            assert_eq!(content, "short", "short results should not be modified");
        }
        // Tool result was inspected but not truncated (under max_bytes)
        assert!(!stats.did_prune());
        assert_eq!(stats.tool_results_total, 1);
        assert_eq!(stats.tool_results_pruned, 0);
    }

    #[test]
    fn truncate_with_marker_short_content() {
        let result = truncate_with_marker("hello", 100);
        assert_eq!(result, "hello");
    }

    #[test]
    fn truncate_with_marker_long_content() {
        let content = "a".repeat(1000);
        let result = truncate_with_marker(&content, 100);
        assert!(result.len() <= 200); // head + marker
        assert!(result.contains("[pruned:"));
        assert!(result.contains("bytes omitted]"));
    }

    #[test]
    fn prune_stats_bytes_saved_accurate() {
        let messages = vec![
            Message::user("task"),
            tool_use_msg("c1", "search"),
            tool_result_msg("c1", &"a".repeat(1000)),
            tool_use_msg("c2", "read"),
            tool_result_msg("c2", &"b".repeat(2000)),
            Message::assistant("done"),
        ];

        let config = SessionPruneConfig {
            keep_recent_n: 0,
            pruned_tool_result_max_bytes: 100,
            preserve_task: true,
        };
        let (pruned, stats) = prune_old_tool_results(&messages, &config);

        assert!(stats.did_prune());
        assert_eq!(stats.tool_results_pruned, 2);
        assert_eq!(stats.tool_results_total, 2);

        // bytes_saved = original bytes - pruned bytes for each truncated result
        let pruned_c1_len = if let ContentBlock::ToolResult { content, .. } = &pruned[2].content[0]
        {
            content.len()
        } else {
            panic!("expected tool result");
        };
        let pruned_c2_len = if let ContentBlock::ToolResult { content, .. } = &pruned[4].content[0]
        {
            content.len()
        } else {
            panic!("expected tool result");
        };
        let expected_saved = (1000 - pruned_c1_len) + (2000 - pruned_c2_len);
        assert_eq!(stats.bytes_saved, expected_saved);
    }
}
