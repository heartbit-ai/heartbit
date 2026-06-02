//! Rendering: the immediate-mode view of [`App`] into a ratatui [`Frame`]. The
//! transcript flattening is verified with a `TestBackend` render test.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use crate::app::App;
use crate::cells::Cell;

const SPINNER: [char; 4] = ['⠋', '⠙', '⠹', '⠸'];

/// Flatten the transcript (history + the live streaming reply) into lines.
pub fn transcript_lines(app: &App) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for cell in &app.history {
        lines.extend(cell.to_lines());
        lines.push(Line::raw(""));
    }
    if let Some(active) = &app.active {
        lines.extend(Cell::Agent(active.clone()).to_lines());
    }
    lines
}

/// Draw the whole UI: transcript (top), status line, composer (bottom), and the
/// approval modal overlay when present.
pub fn view(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let comp_lines = app.composer.render_lines().len().max(1) as u16;
    let comp_h = (comp_lines + 2).clamp(3, 8);
    let chunks = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(comp_h),
    ])
    .split(area);

    // --- transcript ---
    let lines = transcript_lines(app);
    let total = lines.len() as u16;
    let view_h = chunks[0].height;
    let max_off = total.saturating_sub(view_h);
    let offset = max_off.saturating_sub(app.scroll);
    let transcript = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((offset, 0));
    frame.render_widget(transcript, chunks[0]);

    // --- status line ---
    let state = if app.running {
        format!("{} working", SPINNER[app.spinner % SPINNER.len()])
    } else {
        "ready".to_string()
    };
    let status = Line::from(vec![
        Span::styled(
            format!(" {} ", app.model),
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(
                "· {}+{} tok ",
                app.tokens.input_tokens, app.tokens.output_tokens
            ),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(
            format!("· {state} "),
            Style::default().fg(if app.running {
                Color::Yellow
            } else {
                Color::Green
            }),
        ),
    ]);
    frame.render_widget(Paragraph::new(status), chunks[1]);

    // --- composer ---
    let comp_text: Vec<Line> = if app.composer.is_empty() && app.modal.is_none() {
        vec![Line::from(Span::styled(
            "Type a message and press Enter…",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        app.composer
            .render_lines()
            .into_iter()
            .map(Line::raw)
            .collect()
    };
    let title = if app.modal.is_some() {
        " approval pending — answer the prompt ".to_string()
    } else {
        " Enter send · Shift+Enter newline · ↑↓ history · Ctrl+C quit ".to_string()
    };
    let composer =
        Paragraph::new(comp_text).block(Block::default().borders(Borders::ALL).title(title));
    frame.render_widget(composer, chunks[2]);

    // --- approval modal overlay ---
    if let Some(modal) = &app.modal {
        let w = area.width.min(72);
        let h = (modal.tools.len() as u16 + 5).min(area.height);
        let rect = centered(area, w, h);
        frame.render_widget(Clear, rect);
        let mut mlines = vec![Line::from(Span::styled(
            "The agent wants to run:",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ))];
        for t in &modal.tools {
            let summary: String = t.input.chars().take(48).collect();
            mlines.push(Line::raw(format!("  • {}  {}", t.name, summary)));
        }
        mlines.push(Line::raw(""));
        mlines.push(Line::from(Span::styled(
            "[y] allow   [a] always allow   [n] deny",
            Style::default().fg(Color::Cyan),
        )));
        let modal_widget = Paragraph::new(mlines)
            .block(Block::default().borders(Borders::ALL).title(" approve? "))
            .wrap(Wrap { trim: false });
        frame.render_widget(modal_widget, rect);
    } else {
        // Show the text cursor in the composer when no modal is up.
        let (crow, ccol) = app.composer.cursor();
        frame.set_cursor_position(Position::new(
            chunks[2].x + 1 + ccol as u16,
            chunks[2].y + 1 + crow as u16,
        ));
    }
}

fn centered(area: Rect, w: u16, h: u16) -> Rect {
    Rect {
        x: area.x + area.width.saturating_sub(w) / 2,
        y: area.y + area.height.saturating_sub(h) / 2,
        width: w.min(area.width),
        height: h.min(area.height),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::buffer::Buffer;

    fn buffer_text(buf: &Buffer) -> String {
        let area = *buf.area();
        let mut s = String::new();
        for y in 0..area.height {
            for x in 0..area.width {
                s.push_str(buf[(x, y)].symbol());
            }
            s.push('\n');
        }
        s
    }

    #[test]
    fn renders_transcript_status_and_composer() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("qwen-235b");
        app.history.push(Cell::User("hello world".into()));
        app.active = Some("hi there".into());
        app.running = true;
        app.composer.insert_str("draft");
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("hello world"), "user msg missing:\n{text}");
        assert!(
            text.contains("hi there"),
            "streaming reply missing:\n{text}"
        );
        assert!(
            text.contains("qwen-235b"),
            "model missing in status:\n{text}"
        );
        assert!(
            text.contains("working"),
            "running indicator missing:\n{text}"
        );
        assert!(text.contains("draft"), "composer text missing:\n{text}");
    }

    #[test]
    fn renders_approval_modal_over_transcript() {
        use crate::msg::PendingTool;
        use std::sync::mpsc::sync_channel;
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        let (tx, _rx) = sync_channel(1);
        app.update(crate::msg::Msg::Approval {
            tools: vec![PendingTool {
                name: "bash".into(),
                input: "rm -rf /tmp/x".into(),
            }],
            reply: tx,
        });
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("approve"), "modal title missing:\n{text}");
        assert!(text.contains("bash"), "tool name missing:\n{text}");
        assert!(text.contains("allow"), "approval options missing:\n{text}");
    }
}
