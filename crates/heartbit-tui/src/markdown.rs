//! Render Markdown agent output into styled ratatui [`Line`]s. Headings, bold,
//! italic, inline code, fenced code blocks, and lists are styled; everything
//! else degrades to readable text. Pure (no I/O) so it is unit-testable.

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// The colour used for inline code and code blocks.
const CODE: Color = Color::Yellow;
/// The colour used for headings.
const HEADING: Color = Color::Cyan;

/// Accumulates Markdown events into styled lines.
struct Renderer {
    lines: Vec<Line<'static>>,
    spans: Vec<Span<'static>>,
    style: Style,
    stack: Vec<Style>,
    /// Per-level ordered-list counters (`None` = unordered).
    list: Vec<Option<u64>>,
    in_code: bool,
}

impl Renderer {
    fn new() -> Self {
        Self {
            lines: Vec::new(),
            spans: Vec::new(),
            style: Style::default(),
            stack: Vec::new(),
            list: Vec::new(),
            in_code: false,
        }
    }

    /// Emit the current spans as a line and start a new one.
    fn flush(&mut self) {
        let spans = std::mem::take(&mut self.spans);
        self.lines.push(Line::from(spans));
    }

    /// Insert a blank separator line (deduped; never leading).
    fn blank(&mut self) {
        if self.lines.is_empty() {
            return;
        }
        if matches!(self.lines.last(), Some(l) if l.spans.is_empty()) {
            return;
        }
        self.lines.push(Line::default());
    }

    fn finish(mut self) -> Vec<Line<'static>> {
        if !self.spans.is_empty() {
            self.flush();
        }
        while matches!(self.lines.first(), Some(l) if l.spans.is_empty()) {
            self.lines.remove(0);
        }
        while matches!(self.lines.last(), Some(l) if l.spans.is_empty()) {
            self.lines.pop();
        }
        self.lines
    }
}

/// Render Markdown `text` into styled ratatui lines.
pub fn render(text: &str) -> Vec<Line<'static>> {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    let mut r = Renderer::new();

    for ev in Parser::new_ext(text, opts) {
        match ev {
            Event::Start(tag) => match tag {
                Tag::Heading { .. } => {
                    if !r.spans.is_empty() {
                        r.flush();
                    }
                    r.blank();
                    r.stack.push(r.style);
                    r.style = Style::default().fg(HEADING).add_modifier(Modifier::BOLD);
                }
                Tag::Strong => {
                    r.stack.push(r.style);
                    r.style = r.style.add_modifier(Modifier::BOLD);
                }
                Tag::Emphasis => {
                    r.stack.push(r.style);
                    r.style = r.style.add_modifier(Modifier::ITALIC);
                }
                Tag::Strikethrough => {
                    r.stack.push(r.style);
                    r.style = r.style.add_modifier(Modifier::CROSSED_OUT);
                }
                Tag::CodeBlock(_) => {
                    if !r.spans.is_empty() {
                        r.flush();
                    }
                    r.in_code = true;
                }
                Tag::List(start) => r.list.push(start),
                Tag::Item => {
                    if !r.spans.is_empty() {
                        r.flush();
                    }
                    let indent = "  ".repeat(r.list.len().saturating_sub(1));
                    let marker = match r.list.last_mut() {
                        Some(Some(n)) => {
                            let m = format!("{indent}{n}. ");
                            *n += 1;
                            m
                        }
                        _ => format!("{indent}• "),
                    };
                    r.spans
                        .push(Span::styled(marker, Style::default().fg(Color::DarkGray)));
                }
                _ => {}
            },
            Event::End(tag) => match tag {
                TagEnd::Heading(_) => {
                    r.flush();
                    r.style = r.stack.pop().unwrap_or_default();
                    r.blank();
                }
                TagEnd::Strong | TagEnd::Emphasis | TagEnd::Strikethrough => {
                    r.style = r.stack.pop().unwrap_or_default();
                }
                TagEnd::CodeBlock => {
                    if !r.spans.is_empty() {
                        r.flush();
                    }
                    r.in_code = false;
                    r.blank();
                }
                TagEnd::Paragraph => {
                    r.flush();
                    if r.list.is_empty() {
                        r.blank();
                    }
                }
                TagEnd::Item => r.flush(),
                TagEnd::List(_) => {
                    r.list.pop();
                    r.blank();
                }
                _ => {}
            },
            Event::Text(t) => {
                if r.in_code {
                    let style = Style::default().fg(CODE);
                    for (i, seg) in t.split('\n').enumerate() {
                        if i > 0 {
                            r.flush();
                        }
                        if !seg.is_empty() {
                            r.spans.push(Span::styled(seg.to_string(), style));
                        }
                    }
                } else {
                    let style = r.style;
                    let s = t.into_string();
                    if !s.is_empty() {
                        r.spans.push(Span::styled(s, style));
                    }
                }
            }
            Event::Code(c) => r
                .spans
                .push(Span::styled(c.into_string(), Style::default().fg(CODE))),
            Event::SoftBreak | Event::HardBreak => r.flush(),
            Event::Rule => {
                if !r.spans.is_empty() {
                    r.flush();
                }
                r.lines.push(Line::from(Span::styled(
                    "─".repeat(40),
                    Style::default().fg(Color::DarkGray),
                )));
            }
            _ => {}
        }
    }
    r.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::text::Span;

    fn all_text(lines: &[Line<'static>]) -> String {
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

    fn span_containing<'a>(lines: &'a [Line<'static>], needle: &str) -> Option<&'a Span<'static>> {
        lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .find(|s| s.content.contains(needle))
    }

    #[test]
    fn plain_text_passthrough() {
        assert_eq!(all_text(&render("hello world")).trim(), "hello world");
    }

    #[test]
    fn heading_is_bold_and_drops_hashes() {
        let lines = render("# Title");
        let s = span_containing(&lines, "Title").expect("heading text");
        assert!(
            s.style.add_modifier.contains(Modifier::BOLD),
            "heading must be bold"
        );
        assert_eq!(s.style.fg, Some(HEADING), "heading must be coloured");
        assert!(
            !all_text(&lines).contains('#'),
            "the # marker must be dropped"
        );
    }

    #[test]
    fn strong_is_bold() {
        let lines = render("**hi**");
        let s = span_containing(&lines, "hi").expect("strong text");
        assert!(s.style.add_modifier.contains(Modifier::BOLD));
        assert!(
            !all_text(&lines).contains('*'),
            "** markers must be dropped"
        );
    }

    #[test]
    fn emphasis_is_italic() {
        let lines = render("*hi*");
        let s = span_containing(&lines, "hi").expect("emphasis text");
        assert!(s.style.add_modifier.contains(Modifier::ITALIC));
    }

    #[test]
    fn inline_code_is_coloured() {
        let lines = render("use `cargo build` now");
        let s = span_containing(&lines, "cargo build").expect("inline code");
        assert_eq!(s.style.fg, Some(CODE));
        assert!(!all_text(&lines).contains('`'), "backticks must be dropped");
    }

    #[test]
    fn code_block_drops_fences_and_styles() {
        let lines = render("```rust\nfn main() {}\n```");
        let s = span_containing(&lines, "fn main()").expect("code line");
        assert_eq!(s.style.fg, Some(CODE));
        assert!(!all_text(&lines).contains("```"), "fences must be dropped");
    }

    #[test]
    fn bullet_list_has_markers() {
        let lines = render("- alpha\n- beta");
        let text = all_text(&lines);
        assert!(text.contains('•'), "bullet marker missing: {text}");
        assert!(text.contains("alpha") && text.contains("beta"));
    }

    #[test]
    fn agent_pattern_strips_all_syntax() {
        // A realistic answer: heading, bold, italic, list, fenced code. The final
        // render must contain NO raw Markdown markers (streaming may show partial
        // markers transiently, but the settled text is clean).
        let md = "# Rust Ownership\n**Memory Management** without GC:\n\n- alpha\n- beta\n\n\
                  You can *move* or *borrow* it.\n\n```rust\nlet s = 1;\n```";
        let text = all_text(&render(md));
        assert!(!text.contains('#'), "heading marker leaked: {text}");
        assert!(!text.contains('*'), "bold/italic marker leaked: {text}");
        assert!(!text.contains("```"), "code fence leaked: {text}");
        assert!(
            text.contains("Memory Management")
                && text.contains("move")
                && text.contains("borrow")
                && text.contains("let s = 1;"),
            "content lost: {text}"
        );
    }

    #[test]
    fn soft_break_is_a_line_break() {
        // A single newline inside a paragraph is preserved as a line break.
        let lines = render("alpha\nbeta");
        let content: Vec<String> = lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .filter(|s| !s.trim().is_empty())
            .collect();
        assert_eq!(content, vec!["alpha".to_string(), "beta".to_string()]);
    }
}
