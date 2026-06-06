//! Transcript cells and their rendering to ratatui [`Line`]s. Pure (no terminal),
//! so the visual structure is unit-testable with plain assertions.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use serde::{Deserialize, Serialize};

/// Lifecycle of a tool call in the transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolStatus {
    Running,
    Ok,
    Failed,
}

/// One entry in the conversation transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        /// The sub-agent that ran it (multi-agent mode) — rendered as a colored
        /// badge so the transcript shows who did what. `None` in single mode.
        agent: Option<String>,
    },
    /// A small framework notice (guardrail / retry / compaction / error).
    Notice(String),
    /// The model's chain-of-thought (reasoning models only) — rendered dimmed
    /// and distinct from the answer.
    Reasoning(String),
    /// A `/stats` card: the computed trace summary, rendered styled (human
    /// units, sparkline, error highlighting). Boxed to keep `Cell` small.
    Stats {
        label: String,
        stats: Box<crate::trace_stats::TraceStats>,
    },
}

/// `982` · `4.4k` · `1.2M` — compact token counts.
fn fmt_tokens(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    }
}

/// `2ms` · `19.8s` · `1m47s` — human durations.
fn fmt_ms(ms: u64) -> String {
    if ms < 1_000 {
        format!("{ms}ms")
    } else if ms < 60_000 {
        format!("{:.1}s", ms as f64 / 1_000.0)
    } else {
        format!("{}m{}s", ms / 60_000, (ms % 60_000) / 1_000)
    }
}

/// Downsample to ≤ `width` buckets (mean), scale to the max, render ▁▂▃▄▅▆▇█.
fn sparkline(values: &[u64], width: usize) -> String {
    const GLYPHS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if values.is_empty() || width == 0 {
        return String::new();
    }
    let chunk = values.len().div_ceil(width);
    let buckets: Vec<u64> = values
        .chunks(chunk)
        .map(|c| c.iter().sum::<u64>() / c.len() as u64)
        .collect();
    let max = buckets.iter().copied().max().unwrap_or(0).max(1);
    buckets
        .iter()
        .map(|&v| GLYPHS[((v * 7) / max) as usize])
        .collect()
}

/// Max diff lines shown inline before truncation (compact "aperçu" philosophy).
const MAX_DIFF_LINES: usize = 10;

/// Render a tool's input as colored diff `Line`s (red `-` / green `+` / dim
/// context), capped at `max` with a "… N more" note. Empty for non-editing
/// tools or malformed input — the caller then shows only the compact header.
/// Shared by the transcript tool cell and the approval modal.
pub fn diff_preview(tool_name: &str, input: &str, max: usize) -> Vec<Line<'static>> {
    let diff = crate::diff::diff_lines(tool_name, input);
    if diff.is_empty() {
        return Vec::new();
    }
    let total = diff.len();
    let mut out: Vec<Line<'static>> = diff
        .iter()
        .take(max)
        .map(|d| {
            let (sign, style) = match d.kind {
                crate::diff::DiffKind::Add => ("+", Style::default().fg(Color::Green)),
                crate::diff::DiffKind::Del => ("-", Style::default().fg(Color::Red)),
                crate::diff::DiffKind::Ctx => (" ", Style::default().fg(Color::DarkGray)),
            };
            Line::from(Span::styled(format!("  {sign}{}", d.text), style))
        })
        .collect();
    if total > max {
        out.push(Line::from(Span::styled(
            format!("  … ({} more diff lines)", total - max),
            Style::default().fg(Color::DarkGray),
        )));
    }
    out
}

/// A stable identity color for an agent name (consistent across the transcript
/// and the roster panel) — hashed into a small distinct palette (AgentPipe-style).
pub fn agent_color(name: &str) -> Color {
    const PALETTE: [Color; 6] = [
        Color::Cyan,
        Color::Magenta,
        Color::Green,
        Color::Blue,
        Color::LightYellow,
        Color::LightRed,
    ];
    let h = name
        .bytes()
        .fold(0u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32));
    PALETTE[(h as usize) % PALETTE.len()]
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
            // The assistant answers in Markdown — render it styled (headings,
            // bold, code, lists) instead of showing raw syntax.
            Cell::Agent(text) => crate::markdown::render(text),
            Cell::Tool {
                name,
                input,
                status,
                output,
                duration_ms,
                agent,
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
                let mut spans = Vec::new();
                // Multi-agent: a colored per-agent badge so you see WHO ran it.
                if let Some(a) = agent {
                    spans.push(Span::styled(
                        format!("{a} "),
                        Style::default()
                            .fg(agent_color(a))
                            .add_modifier(Modifier::BOLD),
                    ));
                }
                spans.push(Span::styled(
                    format!("{marker} {name}{timing}"),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ));
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
                let mut lines = vec![Line::from(spans)];
                // For file-editing tools, render the change as a compact, capped
                // colored diff under the header (the SOTA "see what changed").
                lines.extend(diff_preview(name, input, MAX_DIFF_LINES));
                lines
            }
            Cell::Notice(text) => vec![Line::from(Span::styled(
                format!("— {text}"),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            ))],
            // Chain-of-thought: dimmed + italic, with a "thinking" header, so it
            // reads as the model's scratchpad — clearly not the answer.
            Cell::Reasoning(text) => {
                let dim = Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC);
                let mut lines = vec![Line::from(Span::styled(
                    "💭 thinking",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::ITALIC),
                ))];
                lines.extend(
                    text.split('\n')
                        .map(|l| Line::from(Span::styled(format!("  {l}"), dim))),
                );
                lines
            }
            Cell::Stats { label, stats } => stats_card(label, stats),
        }
    }
}

/// Build the `/stats` card: colored section rows, human units, a context
/// sparkline, red error highlighting. Pure — fully covered by unit tests.
fn stats_card(label: &str, s: &crate::trace_stats::TraceStats) -> Vec<Line<'static>> {
    let dim = Style::default().fg(Color::DarkGray);
    let title = Style::default()
        .fg(Color::Magenta)
        .add_modifier(Modifier::BOLD);
    let lbl = |name: &str| Span::styled(format!("  {name:<10} "), dim);
    let mut lines = vec![
        Line::from(vec![
            Span::styled("▎ ", title),
            Span::styled(format!("stats — session {label}"), title),
            Span::styled(
                format!(" · {} · {} llm calls", fmt_ms(s.duration_ms), s.llm_calls),
                dim,
            ),
        ]),
        Line::raw(""),
    ];
    // tokens — cache % over total input (0% is still signal: caching is off).
    let cache_pct = s.total_cache_read_tokens * 100 / s.total_input_tokens.max(1);
    lines.push(Line::from(vec![
        lbl("tokens"),
        Span::raw(format!(
            "in {} · out {} · cache {cache_pct}%",
            fmt_tokens(s.total_input_tokens),
            fmt_tokens(s.total_output_tokens)
        )),
    ]));
    // context growth — needs at least two calls to be a curve.
    if s.turn_input_tokens.len() >= 2 {
        let first = *s.turn_input_tokens.first().unwrap_or(&0);
        let last = *s.turn_input_tokens.last().unwrap_or(&0);
        let max = s.turn_input_tokens.iter().copied().max().unwrap_or(0);
        lines.push(Line::from(vec![
            lbl("context"),
            Span::raw(format!(
                "{} → {} /call (max {})  ",
                fmt_tokens(first),
                fmt_tokens(last),
                fmt_tokens(max)
            )),
            Span::styled(sparkline(&s.turn_input_tokens, 24), dim),
        ]));
    }
    lines.push(Line::from(vec![
        lbl("latency"),
        Span::raw(format!(
            "llm p50 {} p95 {} · ttft {}/{}",
            fmt_ms(s.llm_latency_p50_ms),
            fmt_ms(s.llm_latency_p95_ms),
            fmt_ms(s.ttft_p50_ms),
            fmt_ms(s.ttft_p95_ms)
        )),
    ]));
    let failed_style = if s.run_failed > 0 {
        Style::default().fg(Color::Red)
    } else {
        Style::default()
    };
    lines.push(Line::from(vec![
        lbl("runs"),
        Span::raw(format!("{} ok · ", s.run_completed)),
        Span::styled(format!("{} failed", s.run_failed), failed_style),
    ]));
    // friction — green "none", or the non-zero counters in yellow.
    let friction: Vec<String> = [
        (s.retries, "retries"),
        (s.doom_loops, "doom-loops"),
        (s.guardrail_denied, "guardrail-denied"),
        (s.guardrail_warned, "guardrail-warned"),
        (s.interrupts, "interrupts"),
        (s.compactions, "compactions"),
        (s.prunes, "prunes"),
    ]
    .iter()
    .filter(|(n, _)| *n > 0)
    .map(|(n, name)| format!("{name} {n}"))
    .collect();
    lines.push(Line::from(vec![
        lbl("friction"),
        if friction.is_empty() {
            Span::styled("none".to_string(), Style::default().fg(Color::Green))
        } else {
            Span::styled(friction.join(" · "), Style::default().fg(Color::Yellow))
        },
    ]));
    let human = if s.approval_mean_latency_ms < 100 {
        "instant".to_string()
    } else {
        fmt_ms(s.approval_mean_latency_ms)
    };
    lines.push(Line::from(vec![
        lbl("approvals"),
        Span::raw(format!(
            "{} ({} denied) · {human}",
            s.approvals, s.approval_denials
        )),
    ]));
    if !s.tools.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            format!(
                "  {:<14} {:<5} {:<6} {:<8} {}",
                "tool", "×", "err", "p50", "p95"
            ),
            dim,
        )));
        for (name, t) in &s.tools {
            let err = if t.errors > 0 {
                Span::styled(
                    format!("{:<6} ", format!("{} ⚠", t.errors)),
                    Style::default().fg(Color::Red),
                )
            } else {
                Span::styled(format!("{:<7}", "—"), dim)
            };
            lines.push(Line::from(vec![
                Span::raw(format!("  {:<14} {:<5} ", name, t.count)),
                err,
                Span::raw(format!("{:<8} {}", fmt_ms(t.p50_ms), fmt_ms(t.p95_ms))),
            ]));
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_units_format() {
        assert_eq!(fmt_tokens(982), "982");
        assert_eq!(fmt_tokens(4_400), "4.4k");
        assert_eq!(fmt_tokens(1_200_000), "1.2M");
        assert_eq!(fmt_ms(2), "2ms");
        assert_eq!(fmt_ms(355), "355ms");
        assert_eq!(fmt_ms(19_822), "19.8s");
        assert_eq!(fmt_ms(71_100), "1m11s");
        assert_eq!(fmt_ms(107_000), "1m47s");
    }

    #[test]
    fn sparkline_scales_and_downsamples() {
        assert_eq!(sparkline(&[0, 7], 8), "▁█");
        let s = sparkline(&(0..100u64).collect::<Vec<_>>(), 8);
        assert_eq!(s.chars().count(), 8, "{s}");
        assert!(s.ends_with('█'), "{s}");
        assert_eq!(sparkline(&[], 8), "");
    }

    #[test]
    fn stats_cell_renders_the_card() {
        let mut stats = crate::trace_stats::TraceStats {
            llm_calls: 21,
            duration_ms: 107_000,
            total_input_tokens: 86_700,
            total_output_tokens: 6_900,
            total_cache_read_tokens: 35_000,
            turn_input_tokens: vec![4_400, 5_000, 7_100, 13_250, 7_126],
            run_completed: 8,
            ..Default::default()
        };
        stats.tools.insert(
            "webfetch".into(),
            crate::trace_stats::ToolStat {
                count: 11,
                errors: 2,
                p50_ms: 355,
                p95_ms: 2_957,
                ..Default::default()
            },
        );
        let cell = Cell::Stats {
            label: "abc-1".into(),
            stats: Box::new(stats),
        };
        let text = plain(&cell.to_lines());
        assert!(text.contains("stats — session abc-1"), "{text}");
        assert!(text.contains("21 llm calls"), "{text}");
        assert!(text.contains("86.7k"), "{text}");
        assert!(text.contains("cache 40%"), "{text}");
        assert!(text.contains("webfetch"), "{text}");
        assert!(text.contains("2 ⚠"), "{text}");
        assert!(
            text.contains('█') || text.contains('▁'),
            "sparkline:\n{text}"
        );
        assert!(text.contains("8 ok"), "{text}");
        assert!(text.contains("none"), "friction none:\n{text}");
    }

    #[test]
    fn stats_cell_skips_sparkline_under_two_calls() {
        let stats = crate::trace_stats::TraceStats {
            turn_input_tokens: vec![4_400],
            ..Default::default()
        };
        let text = plain(
            &Cell::Stats {
                label: "x".into(),
                stats: Box::new(stats),
            }
            .to_lines(),
        );
        assert!(!text.contains("context"), "{text}");
    }

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
    fn reasoning_cell_renders_dimmed_with_header() {
        let lines = Cell::Reasoning("step one\nstep two".into()).to_lines();
        // A "thinking" header plus one line per reasoning line.
        assert_eq!(lines.len(), 3);
        assert!(plain(&lines).contains("thinking"));
        assert!(plain(&lines).contains("step one"));
        assert!(plain(&lines).contains("step two"));
        // The body lines must be dimmed + italic (clearly not the answer).
        let body = &lines[1];
        assert!(
            body.spans
                .iter()
                .all(|s| s.style.add_modifier.contains(Modifier::ITALIC)
                    && s.style.fg == Some(Color::DarkGray)),
            "reasoning body must be dim italic"
        );
    }

    #[test]
    fn agent_cell_renders_markdown_styled() {
        let lines = Cell::Agent("# Heading\n\nsome **bold** text".into()).to_lines();
        let heading = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .find(|s| s.content.contains("Heading"));
        assert!(
            heading
                .map(|s| s.style.add_modifier.contains(Modifier::BOLD))
                .unwrap_or(false),
            "an agent Markdown heading must render bold"
        );
        let text = plain(&lines);
        assert!(
            !text.contains('#') && !text.contains('*'),
            "raw Markdown syntax must not be shown: {text}"
        );
    }

    #[test]
    fn running_tool_shows_name_and_marker() {
        let cell = Cell::Tool {
            name: "bash".into(),
            input: "{\"command\":\"ls\"}".into(),
            status: ToolStatus::Running,
            output: None,
            duration_ms: None,
            agent: None,
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
            agent: None,
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
            agent: None,
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
            agent: None,
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
            agent: None,
        };
        let s = plain(&cell.to_lines());
        assert!(s.contains("ls -la /tmp"), "command not summarised: {s}");
        assert!(!s.contains('{'), "raw JSON leaked into the summary: {s}");
    }

    #[test]
    fn edit_tool_cell_renders_a_colored_diff() {
        let cell = Cell::Tool {
            name: "edit".into(),
            input: r#"{"file_path":"f.rs","old_string":"let x = 1;","new_string":"let x = 2;"}"#
                .into(),
            status: ToolStatus::Ok,
            output: Some("ok".into()),
            duration_ms: Some(3),
            agent: None,
        };
        let s = plain(&cell.to_lines());
        assert!(s.contains("-let x = 1;"), "removed line missing:\n{s}");
        assert!(s.contains("+let x = 2;"), "added line missing:\n{s}");
        // the colors are set (red del / green add)
        let spans: Vec<_> = cell.to_lines().into_iter().flat_map(|l| l.spans).collect();
        assert!(
            spans
                .iter()
                .any(|sp| sp.content.contains("-let x = 1;") && sp.style.fg == Some(Color::Red))
        );
        assert!(
            spans
                .iter()
                .any(|sp| sp.content.contains("+let x = 2;") && sp.style.fg == Some(Color::Green))
        );
    }

    #[test]
    fn long_diff_is_capped_with_more_note() {
        let new: String = (0..30).map(|i| format!("line {i}\n")).collect();
        let cell = Cell::Tool {
            name: "write".into(),
            input: serde_json::json!({"file_path": "f", "content": new}).to_string(),
            status: ToolStatus::Ok,
            output: None,
            duration_ms: None,
            agent: None,
        };
        let s = plain(&cell.to_lines());
        assert!(s.contains("more diff lines"), "cap note missing:\n{s}");
    }

    #[test]
    fn non_editing_tool_cell_has_no_diff() {
        let cell = Cell::Tool {
            name: "bash".into(),
            input: r#"{"command":"ls"}"#.into(),
            status: ToolStatus::Ok,
            output: Some("a\nb".into()),
            duration_ms: Some(1),
            agent: None,
        };
        let s = plain(&cell.to_lines());
        assert!(
            !s.contains('+') && !s.contains('-'),
            "no diff for bash:\n{s}"
        );
    }

    #[test]
    fn notice_cell_is_dim_dashed() {
        let lines = Cell::Notice("auto-compacted".into()).to_lines();
        assert!(plain(&lines).contains("auto-compacted"));
        assert!(plain(&lines).starts_with("— "));
    }
}
