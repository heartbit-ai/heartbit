//! Rendering: the immediate-mode view of [`App`] into a ratatui [`Frame`]. The
//! transcript flattening is verified with a `TestBackend` render test.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use crate::app::{App, Modal};

const SPINNER: [char; 4] = ['⠋', '⠙', '⠹', '⠸'];

/// Flatten the transcript (history + the live streaming reply) into lines.
pub fn transcript_lines(app: &App) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for cell in &app.history {
        lines.extend(cell.to_lines());
        lines.push(Line::raw(""));
    }
    if let Some(active) = &app.active {
        // The streaming reply renders PLAIN — Markdown is applied only when the
        // cell finalizes into history, so partial markup (e.g. an unclosed `**`)
        // never flashes styled mid-stream.
        lines.extend(active.split('\n').map(|l| Line::raw(l.to_string())));
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
    let view_h = chunks[0].height;
    let transcript = Paragraph::new(lines).wrap(Wrap { trim: false });
    // Count VISUAL rows (post-wrap), not logical lines: a long line occupies
    // several rows, so a bottom-anchored offset computed from `lines.len()`
    // under-scrolls and clips the newest content below the fold. `line_count`
    // uses the same WordWrapper as the renderer (honours `trim: false`); the
    // transcript has no block/border so the border caveat (ratatui #1233) is moot.
    let total = transcript.line_count(chunks[0].width) as u16;
    let max_off = total.saturating_sub(view_h);
    let offset = max_off.saturating_sub(app.scroll);
    frame.render_widget(transcript.scroll((offset, 0)), chunks[0]);

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

    // --- slash-command autocomplete menu (floats above the composer) ---
    let candidates = app.command_candidates();
    if !candidates.is_empty() {
        let h = (candidates.len() as u16 + 2).min(area.height);
        let w = 52.min(area.width);
        let rect = Rect {
            x: chunks[2].x,
            y: chunks[2].y.saturating_sub(h),
            width: w,
            height: h,
        };
        frame.render_widget(Clear, rect);
        let sel = app.menu_selected.min(candidates.len() - 1);
        let items: Vec<Line> = candidates
            .iter()
            .enumerate()
            .map(|(i, (name, desc))| {
                let (name_style, desc_style) = if i == sel {
                    let base = Style::default().fg(Color::Black).bg(Color::Cyan);
                    (base.add_modifier(Modifier::BOLD), base)
                } else {
                    (
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                        Style::default().fg(Color::DarkGray),
                    )
                };
                Line::from(vec![
                    Span::styled(format!(" {name:<7}"), name_style),
                    Span::styled(format!("  {desc} "), desc_style),
                ])
            })
            .collect();
        let menu = Paragraph::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" commands · ↑↓ Tab Enter "),
        );
        frame.render_widget(menu, rect);
    }

    // --- modal overlays ---
    match &app.modal {
        Some(Modal::Approval(modal)) => {
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
        }
        Some(Modal::KeyEntry(modal)) => {
            let w = area.width.min(72);
            let h = 7u16.min(area.height);
            let rect = centered(area, w, h);
            frame.render_widget(Clear, rect);
            let masked: String = "•".repeat(modal.input.chars().count());
            let mlines = vec![
                Line::from(Span::raw("Paste your OpenRouter API key:")),
                Line::raw(""),
                Line::from(Span::styled(
                    if masked.is_empty() {
                        "(empty)".to_string()
                    } else {
                        masked
                    },
                    Style::default().fg(Color::Cyan),
                )),
                Line::raw(""),
                Line::from(Span::styled(
                    "Enter to save · Esc to cancel · keys at openrouter.ai/keys",
                    Style::default().fg(Color::DarkGray),
                )),
            ];
            let modal_widget = Paragraph::new(mlines)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" OpenRouter API key "),
                )
                .wrap(Wrap { trim: false });
            frame.render_widget(modal_widget, rect);
        }
        Some(Modal::ModelPicker(picker)) => {
            let w = area.width.min(74);
            let h = area.height.min(20);
            let rect = centered(area, w, h);
            frame.render_widget(Clear, rect);
            let mut mlines = vec![
                Line::from(vec![
                    Span::styled("search: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        picker.query.clone(),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::raw(""),
            ];
            if app.models.is_empty() {
                mlines.push(Line::from(Span::styled(
                    if app.models_loading {
                        "loading models…"
                    } else {
                        "(no models — use /model <name>)"
                    },
                    Style::default().fg(Color::DarkGray),
                )));
            } else {
                let filtered = crate::models::filter_models(&app.models, &picker.query);
                let visible = (h as usize).saturating_sub(6).max(1);
                let sel = picker.selected.min(filtered.len().saturating_sub(1));
                let start = if sel >= visible { sel + 1 - visible } else { 0 };
                for (fi, &idx) in filtered.iter().enumerate().skip(start).take(visible) {
                    let m = &app.models[idx];
                    let ctx = m
                        .context
                        .map(|c| format!("  {}k", c / 1000))
                        .unwrap_or_default();
                    if fi == sel {
                        mlines.push(Line::from(Span::styled(
                            format!(" ▸ {}{ctx} ", m.id),
                            Style::default()
                                .fg(Color::Black)
                                .bg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        )));
                    } else {
                        mlines.push(Line::from(vec![
                            Span::raw(format!("   {}", m.id)),
                            Span::styled(ctx, Style::default().fg(Color::DarkGray)),
                        ]));
                    }
                }
                mlines.push(Line::raw(""));
                mlines.push(Line::from(Span::styled(
                    format!(
                        "{} models · type to filter · ↑↓ · Enter set · Esc",
                        filtered.len()
                    ),
                    Style::default().fg(Color::DarkGray),
                )));
            }
            let widget = Paragraph::new(mlines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" select model "),
            );
            frame.render_widget(widget, rect);
        }
        None => {
            // Show the text cursor in the composer when no modal is up.
            let (crow, ccol) = app.composer.cursor();
            frame.set_cursor_position(Position::new(
                chunks[2].x + 1 + ccol as u16,
                chunks[2].y + 1 + crow as u16,
            ));
        }
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
    use crate::cells::Cell;
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
    fn streaming_renders_plain_finalized_renders_markdown() {
        fn joined(lines: &[Line<'static>]) -> String {
            lines
                .iter()
                .flat_map(|l| l.spans.iter())
                .map(|s| s.content.as_ref())
                .collect()
        }
        // Streaming `active`: raw Markdown shown as-is (no transient half-parsed markup).
        let mut app = App::new("m");
        app.active = Some("# Heading **bold**".into());
        assert!(
            joined(&transcript_lines(&app)).contains("# Heading **bold**"),
            "streaming text must render plain"
        );
        // Finalized agent cell: Markdown rendered (markers gone).
        let mut app2 = App::new("m");
        app2.history.push(Cell::Agent("# Heading **bold**".into()));
        let t = joined(&transcript_lines(&app2));
        assert!(
            !t.contains('#') && !t.contains('*'),
            "finalized cell must render markdown: {t}"
        );
    }

    #[test]
    fn newest_content_stays_visible_when_earlier_lines_wrap() {
        // Earlier cells have long lines that wrap into many VISUAL rows in a
        // narrow viewport. The transcript must auto-scroll by visual rows so the
        // newest cell is visible — counting logical lines under-scrolls and
        // clips the newest content below the fold (the reported bug).
        let backend = TestBackend::new(20, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        for i in 0..8 {
            app.history.push(Cell::Agent(format!(
                "filler {i} a rather long assistant line that wraps across several visual rows in a narrow viewport indeed"
            )));
        }
        app.history.push(Cell::Agent("NEWEST_MARKER".into()));
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            text.contains("NEWEST_MARKER"),
            "newest content was clipped below the fold:\n{text}"
        );
    }

    #[test]
    fn slash_renders_command_menu() {
        let backend = TestBackend::new(60, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.composer.insert_str("/");
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("commands"), "menu title missing:\n{text}");
        assert!(
            text.contains("/model") && text.contains("/mcp"),
            "commands missing:\n{text}"
        );
    }

    #[test]
    fn model_picker_renders_loading_then_list() {
        use crate::app::{Modal, ModelPicker};
        use crate::models::ModelEntry;
        let backend = TestBackend::new(72, 22);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        // loading state
        app.modal = Some(Modal::ModelPicker(ModelPicker::default()));
        app.models_loading = true;
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("select model"), "title missing:\n{text}");
        assert!(text.contains("loading"), "loading state missing:\n{text}");
        // loaded state
        app.models_loading = false;
        app.models = vec![
            ModelEntry {
                id: "anthropic/claude-x".into(),
                name: "Claude".into(),
                context: Some(200000),
            },
            ModelEntry {
                id: "openai/gpt-y".into(),
                name: "GPT".into(),
                context: None,
            },
        ];
        terminal.draw(|f| view(f, &app)).unwrap();
        let text2 = buffer_text(terminal.backend().buffer());
        assert!(
            text2.contains("anthropic/claude-x") && text2.contains("openai/gpt-y"),
            "model list missing:\n{text2}"
        );
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

    #[test]
    fn key_entry_modal_masks_the_secret() {
        use crate::app::{KeyEntryModal, Modal};
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.modal = Some(Modal::KeyEntry(KeyEntryModal {
            input: "sk-or-supersecret".into(),
        }));
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            text.contains("OpenRouter API key"),
            "title missing:\n{text}"
        );
        assert!(text.contains('•'), "masked dots missing:\n{text}");
        assert!(
            !text.contains("sk-or-supersecret"),
            "the raw key must never be rendered:\n{text}"
        );
    }
}
