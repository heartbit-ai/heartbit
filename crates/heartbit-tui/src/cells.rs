//! Transcript cells and their rendering to ratatui [`Line`]s. Pure (no terminal),
//! so the visual structure is unit-testable with plain assertions.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Lifecycle of a tool call in the transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolStatus {
    Running,
    Ok,
    Failed,
}

/// One entry in the conversation transcript.
#[derive(Debug, Clone)]
pub enum Cell {
    /// A user message.
    User(String),
    /// A finalized assistant message.
    Agent(String),
    /// A tool call, possibly still running.
    Tool {
        name: String,
        input: String,
        status: ToolStatus,
        output: Option<String>,
        duration_ms: Option<u64>,
    },
    /// A small framework notice (guardrail / retry / compaction / error).
    Notice(String),
}

fn first_line(s: &str, max: usize) -> String {
    let line = s.lines().next().unwrap_or("");
    if line.chars().count() > max {
        let truncated: String = line.chars().take(max).collect();
        format!("{truncated}…")
    } else {
        line.to_string()
    }
}

/// A compact, human summary of a tool's input for the one-line "Compact" view:
/// the most relevant string value of its JSON (command/query/path/…), or the
/// first string field, falling back to the raw first line. Clamped to width.
fn summarize_input(input: &str) -> String {
    const KEYS: [&str; 8] = [
        "command",
        "query",
        "q",
        "pattern",
        "path",
        "file_path",
        "url",
        "prompt",
    ];
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(input) {
        for k in KEYS {
            if let Some(s) = map.get(k).and_then(|v| v.as_str()) {
                return first_line(s, 64);
            }
        }
        return map
            .values()
            .find_map(|v| v.as_str())
            .map(|s| first_line(s, 64))
            .unwrap_or_default();
    }
    first_line(input, 64)
}

/// A short one-line summary of a tool's output: the content itself when it is a
/// single short line, otherwise a line count (so the cell never becomes a dump).
fn output_summary(out: &str) -> String {
    let nonempty: Vec<&str> = out
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    match nonempty.as_slice() {
        [] => String::new(),
        [only] if only.chars().count() <= 56 => (*only).to_string(),
        [only] => first_line(only, 56),
        _ => format!("{} lines", out.lines().count()),
    }
}

impl Cell {
    /// Render this cell to styled lines (the caller wraps to width).
    pub fn to_lines(&self) -> Vec<Line<'static>> {
        match self {
            Cell::User(text) => {
                let prefix = Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD);
                text.split('\n')
                    .enumerate()
                    .map(|(i, l)| {
                        if i == 0 {
                            Line::from(vec![Span::styled("› ", prefix), Span::raw(l.to_string())])
                        } else {
                            Line::from(vec![Span::raw("  "), Span::raw(l.to_string())])
                        }
                    })
                    .collect()
            }
            Cell::Agent(text) => text.split('\n').map(|l| Line::raw(l.to_string())).collect(),
            Cell::Tool {
                name,
                input,
                status,
                output,
                duration_ms,
            } => {
                let (marker, color) = match status {
                    ToolStatus::Running => ("⏳", Color::Yellow),
                    ToolStatus::Ok => ("✓", Color::Green),
                    ToolStatus::Failed => ("✗", Color::Red),
                };
                let timing = duration_ms
                    .map(|ms| format!(" ({ms}ms)"))
                    .unwrap_or_default();
                let dim = Style::default().fg(Color::DarkGray);
                // "Compact" preview: one line — marker + name + timing + a short
                // input summary + a short output summary. Never a multi-line dump
                // (the agent sees the full output; the user wants only an aperçu).
                let mut spans = vec![Span::styled(
                    format!("{marker} {name}{timing}"),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                )];
                let in_sum = summarize_input(input);
                if !in_sum.is_empty() {
                    spans.push(Span::styled(format!("  {in_sum}"), dim));
                }
                if let Some(out) = output {
                    let out_sum = output_summary(out);
                    if !out_sum.is_empty() {
                        spans.push(Span::styled(format!("  → {out_sum}"), dim));
                    }
                }
                vec![Line::from(spans)]
            }
            Cell::Notice(text) => vec![Line::from(Span::styled(
                format!("— {text}"),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            ))],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(lines: &[Line<'static>]) -> String {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn user_cell_has_prefix_and_text() {
        let lines = Cell::User("hello".into()).to_lines();
        assert_eq!(lines.len(), 1);
        assert!(plain(&lines).contains("hello"));
        assert!(plain(&lines).starts_with("› "));
    }

    #[test]
    fn user_cell_multiline() {
        let lines = Cell::User("a\nb".into()).to_lines();
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn agent_cell_renders_each_line() {
        let lines = Cell::Agent("line1\nline2".into()).to_lines();
        assert_eq!(lines.len(), 2);
        assert_eq!(plain(&lines), "line1\nline2");
    }

    #[test]
    fn running_tool_shows_name_and_marker() {
        let cell = Cell::Tool {
            name: "bash".into(),
            input: "{\"command\":\"ls\"}".into(),
            status: ToolStatus::Running,
            output: None,
            duration_ms: None,
        };
        let s = plain(&cell.to_lines());
        assert!(s.contains("bash"), "got: {s}");
        assert!(s.contains('⏳'), "running marker missing: {s}");
    }

    #[test]
    fn completed_tool_shows_check_timing_and_output() {
        let cell = Cell::Tool {
            name: "read".into(),
            input: "{}".into(),
            status: ToolStatus::Ok,
            output: Some("file contents".into()),
            duration_ms: Some(42),
        };
        let s = plain(&cell.to_lines());
        assert!(s.contains('✓'));
        assert!(s.contains("42ms"));
        assert!(s.contains("file contents"));
    }

    #[test]
    fn failed_tool_shows_cross() {
        let cell = Cell::Tool {
            name: "verify".into(),
            input: "{}".into(),
            status: ToolStatus::Failed,
            output: Some("boom".into()),
            duration_ms: Some(10),
        };
        assert!(plain(&cell.to_lines()).contains('✗'));
    }

    #[test]
    fn completed_tool_renders_a_single_compact_line() {
        // "Compact" preview: a tool cell is ONE line — marker + name + timing +
        // input summary + a short output summary — never a multi-line dump.
        let out = (0..100)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let cell = Cell::Tool {
            name: "websearch".into(),
            input: r#"{"query":"rust async runtime"}"#.into(),
            status: ToolStatus::Ok,
            output: Some(out),
            duration_ms: Some(1200),
        };
        let lines = cell.to_lines();
        assert_eq!(
            lines.len(),
            1,
            "a tool cell must collapse to one compact line"
        );
        let s = plain(&lines);
        assert!(s.contains("websearch"), "name missing: {s}");
        assert!(s.contains("1200ms"), "timing missing: {s}");
        assert!(
            s.contains("rust async runtime"),
            "input value should be summarised (not raw JSON): {s}"
        );
        assert!(s.contains("100 lines"), "output summary missing: {s}");
    }

    #[test]
    fn compact_tool_input_summary_strips_json_wrapper() {
        // bash's `{"command":"ls -la"}` should read as `ls -la`, not the JSON.
        let cell = Cell::Tool {
            name: "bash".into(),
            input: r#"{"command":"ls -la /tmp"}"#.into(),
            status: ToolStatus::Running,
            output: None,
            duration_ms: None,
        };
        let s = plain(&cell.to_lines());
        assert!(s.contains("ls -la /tmp"), "command not summarised: {s}");
        assert!(!s.contains('{'), "raw JSON leaked into the summary: {s}");
    }

    #[test]
    fn notice_cell_is_dim_dashed() {
        let lines = Cell::Notice("auto-compacted".into()).to_lines();
        assert!(plain(&lines).contains("auto-compacted"));
        assert!(plain(&lines).starts_with("— "));
    }
}
