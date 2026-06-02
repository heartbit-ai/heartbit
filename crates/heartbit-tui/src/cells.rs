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

/// Max lines of tool output shown inline before truncation.
const MAX_OUTPUT_LINES: usize = 12;

fn first_line(s: &str, max: usize) -> String {
    let line = s.lines().next().unwrap_or("");
    if line.chars().count() > max {
        let truncated: String = line.chars().take(max).collect();
        format!("{truncated}…")
    } else {
        line.to_string()
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
                let header = Line::from(vec![
                    Span::styled(
                        format!("{marker} {name}{timing}"),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("  {}", first_line(input, 80)),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]);
                let mut lines = vec![header];
                if let Some(out) = output {
                    let dim = Style::default().fg(Color::DarkGray);
                    let total = out.lines().count();
                    for l in out.lines().take(MAX_OUTPUT_LINES) {
                        lines.push(Line::from(Span::styled(format!("  {l}"), dim)));
                    }
                    if total > MAX_OUTPUT_LINES {
                        lines.push(Line::from(Span::styled(
                            format!("  … ({} more lines)", total - MAX_OUTPUT_LINES),
                            dim,
                        )));
                    }
                }
                lines
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
    fn long_tool_output_is_truncated() {
        let out = (0..100)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let cell = Cell::Tool {
            name: "bash".into(),
            input: "{}".into(),
            status: ToolStatus::Ok,
            output: Some(out),
            duration_ms: Some(1),
        };
        let s = plain(&cell.to_lines());
        assert!(s.contains("more lines"), "should note truncation: {s}");
        // header + MAX_OUTPUT_LINES + truncation note
        assert_eq!(cell.to_lines().len(), 1 + MAX_OUTPUT_LINES + 1);
    }

    #[test]
    fn notice_cell_is_dim_dashed() {
        let lines = Cell::Notice("auto-compacted".into()).to_lines();
        assert!(plain(&lines).contains("auto-compacted"));
        assert!(plain(&lines).starts_with("— "));
    }
}
