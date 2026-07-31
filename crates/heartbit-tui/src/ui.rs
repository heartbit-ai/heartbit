//! Rendering: the immediate-mode view of [`App`] into a ratatui [`Frame`]. The
//! transcript flattening is verified with a `TestBackend` render test.

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use crate::app::{App, Modal};

const SPINNER: [char; 4] = ['⠋', '⠙', '⠹', '⠸'];

/// Height of the queued-messages box above the composer: `0` (no space at
/// all) when nothing is queued, so an idle frame is pixel-identical to one
/// rendered before this feature existed. Otherwise one row per queued entry
/// plus borders, capped so a long backlog can't crowd out the transcript.
fn queue_height(queued_len: usize, frame_h: u16) -> u16 {
    if queued_len == 0 {
        return 0;
    }
    let cap = frame_h / 3;
    ((queued_len as u16).saturating_add(2)).clamp(3, cap.max(3))
}

/// Flatten the transcript (history + the live streaming reply) into lines.
pub fn transcript_lines(app: &App) -> Vec<Line<'static>> {
    // Sweep the Markdown cache for the new frame FIRST (exactly one caller,
    // by design — see `MarkdownCache::begin_frame`'s doc comment): every
    // settled `Cell::Agent` rendered below marks itself touched again this
    // frame, so only cells that scrolled out of the transcript get evicted.
    app.md.begin_frame();
    let mut lines = Vec::new();
    // Fresh session: a small identity header instead of a bare wall of
    // notices (cleared the moment the conversation starts).
    if app.history.is_empty() && app.active.is_none() && app.active_reasoning.is_none() {
        let dim = Style::default().fg(Color::DarkGray);
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            format!("  ♥ heartbit v{}", env!("CARGO_PKG_VERSION")),
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(format!("    {}", app.model), dim)));
        lines.push(Line::from(Span::styled(
            "    /help · Shift+Tab mode · Esc interrupt · @ files · / commands",
            dim,
        )));
        lines.push(Line::raw(""));
    }
    for cell in &app.history {
        // Route the (potentially expensive, syntax-highlighted) Markdown
        // render through the cache; every other cell kind is cheap to build
        // fresh and goes through `to_lines()` unchanged.
        match cell {
            crate::cells::Cell::Agent(text) => lines.extend(app.md.render(text)),
            other => lines.extend(other.to_lines()),
        }
        lines.push(Line::raw(""));
    }
    // Live chain-of-thought (reasoning models) renders dimmed above the answer,
    // matching the finalized `Cell::Reasoning` look, so streaming→settled is
    // seamless.
    if let Some(reasoning) = &app.active_reasoning {
        lines.push(Line::from(Span::styled(
            "💭 thinking",
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::ITALIC),
        )));
        let dim = Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC);
        lines.extend(
            reasoning
                .split('\n')
                .map(|l| Line::from(Span::styled(format!("  {l}"), dim))),
        );
    }
    if let Some(active) = &app.active {
        // Live Markdown while streaming: COMPLETED blocks (blank-line
        // boundary, fence-aware) render styled immediately; only the
        // in-flight tail stays raw — so partial markup (an unclosed `**`)
        // never flashes styled mid-stream.
        lines.extend(crate::markdown::render_streaming(active));
    }
    lines
}

/// Draw the whole UI: transcript (top), status line, composer (bottom), and the
/// approval modal overlay when present.
pub fn view(frame: &mut Frame, app: &App) {
    let area = frame.area();
    // Startup splash: a full-frame overlay that pre-empts EVERYTHING — no
    // transcript, status, composer, or modal renders beneath it. Dismissal
    // (timer / any key) lives in the reducer; this is pure paint.
    if let Some(tick) = app.splash {
        let lines = if area.height < 16 || area.width < 44 {
            vec![Line::from(Span::styled(
                format!("♥ heartbit v{}", env!("CARGO_PKG_VERSION")),
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ))]
        } else {
            crate::splash::splash_lines(tick, &app.model)
        };
        let h = (lines.len() as u16).min(area.height);
        let top = area.height.saturating_sub(h) / 2;
        let rect = Rect {
            x: area.x,
            y: area.y + top,
            width: area.width,
            height: h,
        };
        frame.render_widget(Paragraph::new(lines).alignment(Alignment::Center), rect);
        return;
    }
    // Composer height comes from WRAPPED rows (char-exact, the same math the
    // cursor uses) — logical lines under-count once a line exceeds the inner
    // width, clipping the typed text at the border (the transcript scroll
    // lesson, replayed on input).
    let comp_inner_w = area.width.saturating_sub(2).max(1) as usize;
    let comp_rows = app.composer.wrap_lines(comp_inner_w).len().max(1) as u16;
    let comp_h = (comp_rows + 2).clamp(3, 8);
    // Status spans are built up-front: their measured width decides whether
    // the status bar needs ONE line or TWO (identity row + metrics row) —
    // a long model+advisor pair must not push the state off-screen.
    let (status_identity, status_metrics) = status_spans(app);
    let status_w: usize = status_identity
        .iter()
        .chain(status_metrics.iter())
        .map(|sp: &Span| sp.width())
        .sum();
    let status_h: u16 = if status_w <= area.width as usize {
        1
    } else {
        2
    };
    // Visible input queue (above the composer, T1.3): messages submitted
    // while a turn was in flight. A ZERO-height constraint when the queue is
    // empty (the overwhelmingly common case) means this chunk takes no space
    // at all — the frame below (transcript `line_count`/`max_off`/
    // `scroll_offset`) is untouched, identical to before this feature existed.
    let queue_h = queue_height(app.queued.len(), area.height);
    let chunks = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(status_h),
        Constraint::Length(queue_h),
        Constraint::Length(comp_h),
    ])
    .split(area);

    // --- transcript (+ a right-hand side panel) ---
    // The right column holds the live agent roster (while a multi-agent turn is
    // running) and/or the task list. They STACK vertically inside one fixed-width
    // column — so they never fight over width, and the transcript width (which
    // MUST feed `line_count`, or a stale full width clips the newest line) is
    // unchanged whether one or both panels show.
    // The unified agent delegates as needed — show the roster only when a
    // sub-agent is actually dispatched (any non-Idle row). Every submit seeds
    // the squad as Idle; an idle-only roster on a pure-chat turn would eat a
    // third of the width for nothing (campaign round-1 frame evidence).
    let show_roster = app.running
        && app
            .agents
            .iter()
            .any(|a| !matches!(a.state, crate::app::AgentState::Idle));
    let show_todos = !app.todos.is_empty();
    let want_panel = show_roster || show_todos;
    let (transcript_area, panel_area) = if want_panel && chunks[0].width >= 50 {
        let w = (chunks[0].width / 3).clamp(22, 36);
        let split =
            Layout::horizontal([Constraint::Min(1), Constraint::Length(w)]).split(chunks[0]);
        (split[0], Some(split[1]))
    } else {
        (chunks[0], None)
    };

    let lines = transcript_lines(app);
    let view_h = transcript_area.height;
    let transcript = Paragraph::new(lines).wrap(Wrap { trim: false });
    // Count VISUAL rows (post-wrap) at the TRANSCRIPT width, not logical lines.
    // The usize → u16 narrowing must SATURATE at ratatui's scroll boundary: a
    // wrapping cast past 65,535 rows collapses `max_off` and pins follow mode
    // near the TOP of the transcript (the newest output becomes unreachable).
    let total = transcript
        .line_count(transcript_area.width)
        .min(u16::MAX as usize) as u16;
    let max_off = total.saturating_sub(view_h);
    // Follow-the-bottom by default; top-anchored once the user scrolls up (so
    // streaming output never yanks or drifts the view). `view()` feeds `max_off`
    // back so the wheel handlers can anchor a fresh scroll.
    let offset = app.scroll_offset(max_off);
    frame.render_widget(transcript.scroll((offset, 0)), transcript_area);
    if let Some(rect) = panel_area {
        if show_roster && show_todos {
            // Both visible: roster on top, task list below — stacked in the same
            // column so the task list is visible DURING the run (not only after).
            let halves = Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(rect);
            render_roster(frame, app, halves[0]);
            render_todos(frame, app, halves[1]);
        } else if show_roster {
            render_roster(frame, app, rect);
        } else {
            render_todos(frame, app, rect);
        }
    }

    // --- status bar: identity row (+ metrics row when it would overflow) ---
    if status_h == 1 {
        let mut line = status_identity;
        line.extend(status_metrics);
        frame.render_widget(Paragraph::new(Line::from(line)), chunks[1]);
    } else {
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(status_identity),
                Line::from(status_metrics),
            ]),
            chunks[1],
        );
    }

    // --- queued messages (visible input queue, T1.3) ---
    // Submitted while a turn was in flight: held here (App::queued) instead
    // of the invisible unbounded input channel, so the user can see, edit
    // (↑) and cancel (Esc) them. Nothing renders — and `queue_h` is 0 — while
    // the queue is empty.
    if !app.queued.is_empty() {
        let items: Vec<Line> = app
            .queued
            .iter()
            .map(|q| {
                // The clean DISPLAY text — never the wire payload, which may
                // carry an invisible directive (e.g. Plan mode's prefix).
                let preview = q.display.lines().next().unwrap_or("").trim();
                Line::from(Span::styled(
                    format!(" • {preview}"),
                    Style::default().fg(Color::DarkGray),
                ))
            })
            .collect();
        let widget = Paragraph::new(items)
            .block(Block::default().borders(Borders::ALL).title(format!(
                " queued ({}) · ↑ edit newest · Esc drop ",
                app.queued.len()
            )))
            .wrap(Wrap { trim: true });
        frame.render_widget(widget, chunks[2]);
    }

    // --- composer ---
    // Pre-wrapped rows + a vertical scroll that keeps the cursor's row inside
    // the (height-capped) box: when the draft outgrows the cap, the view
    // follows the cursor instead of clipping it below the border.
    let comp_inner_h = comp_h.saturating_sub(2).max(1) as usize;
    let (cur_row, cur_col) = app.composer.visual_cursor(comp_inner_w);
    let comp_scroll = cur_row.saturating_sub(comp_inner_h - 1);
    let comp_text: Vec<Line> = if app.composer.is_empty() && app.modal.is_none() {
        vec![Line::from(Span::styled(
            "Type a message and press Enter…",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        app.composer
            .wrap_lines(comp_inner_w)
            .into_iter()
            .map(Line::raw)
            .collect()
    };
    let title = match &app.modal {
        // Only the approval modal is actually "pending" — pickers and prompts
        // get a neutral hint (live pty finding: the mode picker showed
        // "approval pending", which lies).
        Some(Modal::Approval(_)) => " approval pending — answer the prompt ".to_string(),
        Some(_) => " answer the prompt above — Esc cancels ".to_string(),
        None => " Enter send · Shift+Enter newline · ↑↓ history · Ctrl+C quit ".to_string(),
    };
    let composer = Paragraph::new(comp_text)
        .scroll((comp_scroll as u16, 0))
        .block(Block::default().borders(Borders::ALL).title(title));
    frame.render_widget(composer, chunks[3]);

    // --- slash-command autocomplete menu (floats above the composer) ---
    let candidates = app.command_candidates();
    if !candidates.is_empty() {
        let h = (candidates.len() as u16 + 2).min(area.height);
        let w = 52.min(area.width);
        let rect = Rect {
            x: chunks[3].x,
            y: chunks[3].y.saturating_sub(h),
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
    } else {
        // --- @-mention file autocomplete (floats above the composer) ---
        let files = app.mention_candidates();
        if !files.is_empty() {
            let h = (files.len() as u16 + 2).min(area.height).min(10);
            let w = 60.min(area.width);
            let rect = Rect {
                x: chunks[3].x,
                y: chunks[3].y.saturating_sub(h),
                width: w,
                height: h,
            };
            frame.render_widget(Clear, rect);
            let sel = app.menu_selected.min(files.len() - 1);
            let items: Vec<Line> = files
                .iter()
                .take(h as usize - 2)
                .enumerate()
                .map(|(i, f)| {
                    if i == sel {
                        Line::from(Span::styled(
                            format!(" ▸ {f} "),
                            Style::default().fg(Color::Black).bg(Color::Cyan),
                        ))
                    } else {
                        Line::from(Span::styled(
                            format!("   {f}"),
                            Style::default().fg(Color::Cyan),
                        ))
                    }
                })
                .collect();
            let menu = Paragraph::new(items).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" files · ↑↓ Tab Enter "),
            );
            frame.render_widget(menu, rect);
        }
    }

    // --- modal overlays ---
    match &app.modal {
        Some(Modal::Question(m)) => {
            let w = area.width.min(84);
            let total = m.request.questions.len();
            let mut mlines: Vec<Line> = Vec::new();
            if let Some(q) = m.request.questions.get(m.current) {
                mlines.push(Line::from(vec![
                    Span::styled(
                        format!(" {} ", q.header),
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("  question {}/{total}", m.current + 1),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
                mlines.push(Line::raw(""));
                mlines.push(Line::from(Span::styled(
                    q.question.clone(),
                    Style::default().add_modifier(Modifier::BOLD),
                )));
                mlines.push(Line::raw(""));
                for (i, opt) in q.options.iter().enumerate() {
                    let mark = if q.multiple {
                        if m.picked.get(i).copied().unwrap_or(false) {
                            "[x]"
                        } else {
                            "[ ]"
                        }
                    } else if i == m.selected {
                        " ▸ "
                    } else {
                        "   "
                    };
                    let style = if i == m.selected {
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    };
                    mlines.push(Line::from(Span::styled(
                        format!("{mark} {}", opt.label),
                        style,
                    )));
                    mlines.push(Line::from(Span::styled(
                        format!("      {}", opt.description),
                        Style::default().fg(Color::DarkGray),
                    )));
                }
                mlines.push(Line::raw(""));
                let hint = if q.multiple {
                    "↑↓ · Space toggle · Enter confirm · Esc dismiss"
                } else {
                    "↑↓ · Enter choose · Esc dismiss"
                };
                mlines.push(Line::from(Span::styled(
                    hint,
                    Style::default().fg(Color::Cyan),
                )));
            }
            let h = (mlines.len() as u16 + 2).min(area.height);
            let rect = centered(area, w, h);
            frame.render_widget(Clear, rect);
            let modal_widget = Paragraph::new(mlines)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" the agent asks "),
                )
                .wrap(Wrap { trim: false });
            frame.render_widget(modal_widget, rect);
        }
        Some(Modal::Approval(modal)) => {
            let w = area.width.min(84);
            let mut mlines = vec![Line::from(Span::styled(
                "The agent wants to run:",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ))];
            for t in &modal.tools {
                mlines.push(Line::from(Span::styled(
                    format!("  • {}", t.name),
                    Style::default().add_modifier(Modifier::BOLD),
                )));
                // For a file edit, show the colored diff so the user reviews the
                // exact change; otherwise show the FULL input (no blind approval).
                let diff = crate::cells::diff_preview(&t.name, &t.input, 16);
                if !diff.is_empty() {
                    mlines.extend(diff);
                } else {
                    for l in t.input.lines().take(8) {
                        mlines.push(Line::from(Span::styled(
                            format!("    {l}"),
                            Style::default().fg(Color::DarkGray),
                        )));
                    }
                }
            }
            mlines.push(Line::raw(""));
            mlines.push(Line::from(Span::styled(
                "[y] allow   [a] always allow   [n] deny   [d] always deny",
                Style::default().fg(Color::Cyan),
            )));
            let h = (mlines.len() as u16 + 2).min(area.height);
            let rect = centered(area, w, h);
            frame.render_widget(Clear, rect);
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
            let title = match picker.target {
                crate::app::ModelTarget::Main => " select model ",
                crate::app::ModelTarget::Advisor => " select advisor model ",
            };
            let widget =
                Paragraph::new(mlines).block(Block::default().borders(Borders::ALL).title(title));
            frame.render_widget(widget, rect);
        }
        Some(Modal::ModePicker { sel }) => {
            let w = area.width.min(74);
            let h = 7u16.min(area.height);
            let rect = centered(area, w, h);
            frame.render_widget(Clear, rect);
            let current = app.permission_mode;
            let mut mlines: Vec<Line> = crate::app::MODES
                .iter()
                .enumerate()
                .map(|(i, m)| {
                    let marker = if *m == current { "●" } else { " " };
                    let row = format!(" {marker} {:<7} {}", m.label(), m.describe());
                    if i == *sel {
                        Line::from(Span::styled(
                            row,
                            Style::default()
                                .fg(Color::Black)
                                .bg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ))
                    } else {
                        Line::raw(row)
                    }
                })
                .collect();
            mlines.push(Line::raw(""));
            mlines.push(Line::from(Span::styled(
                " ↑↓ · Enter set · Esc",
                Style::default().fg(Color::DarkGray),
            )));
            let widget = Paragraph::new(mlines)
                .block(Block::default().borders(Borders::ALL).title(" mode "));
            frame.render_widget(widget, rect);
        }
        Some(Modal::EffortPicker { sel }) => {
            let w = area.width.min(74);
            let h = 8u16.min(area.height);
            let rect = centered(area, w, h);
            frame.render_widget(Clear, rect);
            let current = app.effort;
            let mut mlines: Vec<Line> = crate::app::EffortLevel::ALL
                .iter()
                .enumerate()
                .map(|(i, l)| {
                    let marker = if *l == current { "●" } else { " " };
                    let row = format!(" {marker} {}", l.label());
                    if i == *sel {
                        Line::from(Span::styled(
                            row,
                            Style::default()
                                .fg(Color::Black)
                                .bg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ))
                    } else {
                        Line::raw(row)
                    }
                })
                .collect();
            mlines.push(Line::raw(""));
            mlines.push(Line::from(Span::styled(
                " ↑↓ · Enter set · Esc",
                Style::default().fg(Color::DarkGray),
            )));
            let widget = Paragraph::new(mlines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" reasoning effort "),
            );
            frame.render_widget(widget, rect);
        }
        Some(Modal::HistorySearch(h)) => {
            let matches = app.history_matches(&h.query);
            let w = area.width.min(80);
            let h_box = 9u16.min(area.height);
            let rect = centered(area, w, h_box);
            frame.render_widget(Clear, rect);
            let mut mlines = vec![
                Line::from(vec![
                    Span::styled("(reverse-search) ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        h.query.clone(),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::raw(""),
            ];
            if matches.is_empty() {
                mlines.push(Line::from(Span::styled(
                    "(no matching prompt)",
                    Style::default().fg(Color::DarkGray),
                )));
            } else {
                let sel = h.sel.min(matches.len() - 1);
                for (i, m) in matches.iter().take(4).enumerate() {
                    let line: String = m.replace('\n', " ").chars().take(w as usize - 4).collect();
                    if i == sel {
                        mlines.push(Line::from(Span::styled(
                            format!(" ▸ {line} "),
                            Style::default().fg(Color::Black).bg(Color::Cyan),
                        )));
                    } else {
                        mlines.push(Line::from(Span::styled(
                            format!("   {line}"),
                            Style::default().fg(Color::DarkGray),
                        )));
                    }
                }
            }
            mlines.push(Line::raw(""));
            mlines.push(Line::from(Span::styled(
                "Ctrl+R next · Enter use · Esc cancel",
                Style::default().fg(Color::DarkGray),
            )));
            let widget = Paragraph::new(mlines)
                .block(Block::default().borders(Borders::ALL).title(" history "))
                .wrap(Wrap { trim: false });
            frame.render_widget(widget, rect);
        }
        Some(Modal::SessionPicker(p)) => {
            let w = area.width.min(80);
            let h = (p.sessions.len() as u16 + 4).min(area.height).min(16);
            let rect = centered(area, w, h);
            frame.render_widget(Clear, rect);
            let mut mlines = vec![
                Line::from(Span::styled(
                    "Resume a session:",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::raw(""),
            ];
            let sel = p.sel.min(p.sessions.len().saturating_sub(1));
            for (i, s) in p.sessions.iter().take(h as usize - 4).enumerate() {
                let label: String = s.preview.chars().take(w as usize - 14).collect::<String>();
                let line = format!("{label}  ({} turns)", s.turns);
                if i == sel {
                    mlines.push(Line::from(Span::styled(
                        format!(" ▸ {line} "),
                        Style::default().fg(Color::Black).bg(Color::Cyan),
                    )));
                } else {
                    mlines.push(Line::from(Span::styled(
                        format!("   {line}"),
                        Style::default().fg(Color::DarkGray),
                    )));
                }
            }
            mlines.push(Line::raw(""));
            mlines.push(Line::from(Span::styled(
                "↑↓ select · Enter resume · Esc cancel",
                Style::default().fg(Color::DarkGray),
            )));
            let widget = Paragraph::new(mlines)
                .block(Block::default().borders(Borders::ALL).title(" resume "))
                .wrap(Wrap { trim: false });
            frame.render_widget(widget, rect);
        }
        Some(Modal::HandoffPicker { briefs, sel }) => {
            let w = area.width.min(80);
            let h = (briefs.len() as u16 + 4).min(area.height).min(16);
            let rect = centered(area, w, h);
            frame.render_widget(Clear, rect);
            let mut mlines = vec![
                Line::from(Span::styled(
                    "Seed a session from a handoff brief:",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::raw(""),
            ];
            let sel = (*sel).min(briefs.len().saturating_sub(1));
            for (i, b) in briefs.iter().take(h as usize - 4).enumerate() {
                let label: String = b.preview.chars().take(w as usize - 14).collect();
                let line = format!("{label}  ({})", b.file_name);
                if i == sel {
                    mlines.push(Line::from(Span::styled(
                        format!(" ▸ {line} "),
                        Style::default().fg(Color::Black).bg(Color::Cyan),
                    )));
                } else {
                    mlines.push(Line::from(Span::styled(
                        format!("   {line}"),
                        Style::default().fg(Color::DarkGray),
                    )));
                }
            }
            mlines.push(Line::raw(""));
            mlines.push(Line::from(Span::styled(
                "↑↓ select · Enter seed · Esc cancel",
                Style::default().fg(Color::DarkGray),
            )));
            let widget = Paragraph::new(mlines)
                .block(Block::default().borders(Borders::ALL).title(" handoff "))
                .wrap(Wrap { trim: false });
            frame.render_widget(widget, rect);
        }
        None => {
            // Show the text cursor in the composer when no modal is up — at
            // its WRAPPED position, adjusted by the composer's scroll.
            frame.set_cursor_position(Position::new(
                chunks[3].x + 1 + cur_col as u16,
                chunks[3].y + 1 + (cur_row - comp_scroll) as u16,
            ));
        }
    }
}

/// Build the status-bar spans as two groups: IDENTITY (model · advised by)
/// and METRICS (context · tokens · cache · ttft · scroll · mode · state).
/// The caller renders them on one line when they fit, two otherwise.
fn status_spans(app: &App) -> (Vec<Span<'static>>, Vec<Span<'static>>) {
    let state = if app.running {
        format!("{} working", SPINNER[app.spinner % SPINNER.len()])
    } else {
        "ready".to_string()
    };
    let mut identity = vec![Span::styled(
        format!(" {} ", app.model),
        Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD),
    )];
    // Advisor pairing: keep the judge model permanently visible next to the
    // main model (`/model advisor` to change it; hidden when unset).
    if let Some(advisor) = &app.frontier_model {
        identity.push(Span::styled(
            format!("· advised by {advisor} "),
            Style::default().fg(Color::Magenta),
        ));
    }
    // Reasoning-effort level (`/effort`) — hidden when Off, so the default
    // status line stays identical to before this feature existed.
    if app.effort != crate::app::EffortLevel::Off {
        identity.push(Span::styled(
            format!("· effort:{} ", app.effort.label()),
            Style::default().fg(Color::Magenta),
        ));
    }
    // Context-window fill: a small gauge when we know the model's limit, else a
    // raw token count. Color thresholds: green <70%, yellow <90%, red beyond.
    let mut metrics = context_spans(app);
    metrics.push(Span::styled(
        format!(
            "· {} tok ",
            human_tokens(app.tokens.input_tokens + app.tokens.output_tokens)
        ),
        Style::default().fg(Color::DarkGray),
    ));
    // Cache-hit metric: surface cumulative prompt-cache reads (green = savings).
    if app.tokens.cache_read_input_tokens > 0 {
        metrics.push(Span::styled(
            format!(
                "· {} cached ",
                human_tokens(app.tokens.cache_read_input_tokens)
            ),
            Style::default().fg(Color::Green),
        ));
    }
    if app.last_ttft_ms > 0 {
        metrics.push(Span::styled(
            format!("· {} ttft ", human_ms(app.last_ttft_ms)),
            Style::default().fg(Color::DarkGray),
        ));
    }
    // Un-pinned transcript: say so, and how to get back (streaming keeps
    // appending out of sight — silent drift reads as a hang).
    if !app.follow {
        metrics.push(Span::styled(
            "· ↑ scrolled — wheel down to follow ",
            Style::default().fg(Color::Yellow),
        ));
    }
    // Execution mode — only when not Normal (keeps the line clean).
    if app.permission_mode != crate::app::PermissionMode::Normal {
        let c = match app.permission_mode {
            crate::app::PermissionMode::Yolo => Color::Red,
            crate::app::PermissionMode::Plan => Color::Blue,
            _ => Color::Yellow,
        };
        metrics.push(Span::styled(
            format!("· {} ", app.permission_mode.label()),
            Style::default().fg(c).add_modifier(Modifier::BOLD),
        ));
    }
    metrics.push(Span::styled(
        format!("· {state} "),
        Style::default().fg(if app.running {
            Color::Yellow
        } else {
            Color::Green
        }),
    ));
    (identity, metrics)
}

/// `1234 → "1.2k"`, `999 → "999"` — compact token/number formatting.
fn human_tokens(n: u32) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

/// `1400 → "1.4s"`, `420 → "420ms"` — compact latency formatting.
fn human_ms(ms: u64) -> String {
    if ms >= 1000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        format!("{ms}ms")
    }
}

/// The status-line context indicator: a colored fill gauge `[████░░░░] 42%` when
/// the model's context window is known (catalog loaded), else a raw `12.8k ctx`.
fn context_spans(app: &App) -> Vec<Span<'static>> {
    if app.context_tokens == 0 {
        return Vec::new();
    }
    match app.context_limit() {
        Some(limit) if limit > 0 => {
            let frac = (app.context_tokens as f64 / limit as f64).clamp(0.0, 1.0);
            let pct = (frac * 100.0).round() as u32;
            const W: usize = 8;
            let filled = (frac * W as f64).round() as usize;
            let bar: String = "█".repeat(filled) + &"░".repeat(W - filled);
            let color = if pct < 70 {
                Color::Green
            } else if pct < 90 {
                Color::Yellow
            } else {
                Color::Red
            };
            vec![
                Span::styled(format!("· {bar} "), Style::default().fg(color)),
                Span::styled(format!("{pct}% ctx "), Style::default().fg(color)),
            ]
        }
        _ => vec![Span::styled(
            format!("· {} ctx ", human_tokens(app.context_tokens)),
            Style::default().fg(Color::DarkGray),
        )],
    }
}

/// Render the live task list (mirrors the agent's `todowrite`): a checkbox per
/// task — ✓ done / ⠹ in-progress / ○ pending — with a "N/M done" title.
fn render_todos(frame: &mut Frame, app: &App, area: Rect) {
    use crate::app::TodoStatus;
    let inner_w = area.width.saturating_sub(4) as usize;
    let cap = (area.height as usize).saturating_sub(2).max(1);
    let done = app
        .todos
        .iter()
        .filter(|t| t.status == TodoStatus::Completed)
        .count();
    let spin = SPINNER[app.spinner % SPINNER.len()];
    let mut lines: Vec<Line> = Vec::new();
    for t in app.todos.iter().take(cap) {
        let (icon, style) = match t.status {
            TodoStatus::Completed => (
                "✓".to_string(),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::CROSSED_OUT),
            ),
            TodoStatus::InProgress => (spin.to_string(), Style::default().fg(Color::Yellow)),
            TodoStatus::Pending => ("○".to_string(), Style::default().fg(Color::DarkGray)),
        };
        let text: String = t.content.chars().take(inner_w).collect();
        lines.push(Line::from(vec![
            Span::styled(format!("{icon} "), style),
            Span::styled(text, style),
        ]));
    }
    if app.todos.len() > cap {
        lines.push(Line::from(Span::styled(
            format!("… {} more", app.todos.len() - cap),
            Style::default().fg(Color::DarkGray),
        )));
    }
    let title = format!(" tasks · {done}/{} done ", app.todos.len());
    let panel = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false });
    frame.render_widget(panel, area);
}

/// Render the live agent roster: one row per agent with a state icon (animated
/// spinner while working), its identity color, and a one-line activity. Working
/// agents are listed first so "who is doing what right now" is at a glance.
fn render_roster(frame: &mut Frame, app: &App, area: Rect) {
    use crate::app::AgentState;
    let inner_w = area.width.saturating_sub(2) as usize;
    let mut rows: Vec<&crate::app::AgentRow> = app.agents.iter().collect();
    // Working first, then done/failed, then still-idle (available) — preserving
    // first-seen order within a group.
    rows.sort_by_key(|r| match r.state {
        AgentState::Working => 0,
        AgentState::Done => 1,
        AgentState::Failed => 2,
        AgentState::Idle => 3,
    });
    let spin = SPINNER[app.spinner % SPINNER.len()];
    let mut lines: Vec<Line> = Vec::new();
    for r in rows {
        let (icon, icon_style) = match r.state {
            AgentState::Working => (spin.to_string(), Style::default().fg(Color::Yellow)),
            AgentState::Done => ("✓".to_string(), Style::default().fg(Color::Green)),
            AgentState::Failed => ("✗".to_string(), Style::default().fg(Color::Red)),
            AgentState::Idle => ("·".to_string(), Style::default().fg(Color::DarkGray)),
        };
        let name_style = Style::default()
            .fg(crate::cells::agent_color(&r.name))
            .add_modifier(Modifier::BOLD);
        lines.push(Line::from(vec![
            Span::styled(format!("{icon} "), icon_style),
            Span::styled(r.name.clone(), name_style),
        ]));
        // Activity sub-line (dimmed, indented), plus token cost when finished.
        let detail = if r.state == AgentState::Working {
            r.activity.clone()
        } else if r.tokens > 0 {
            format!("{} · {} tok", r.activity, r.tokens)
        } else {
            r.activity.clone()
        };
        let detail: String = detail.chars().take(inner_w.saturating_sub(3)).collect();
        lines.push(Line::from(Span::styled(
            format!("  {detail}"),
            Style::default().fg(Color::DarkGray),
        )));
    }
    let working = app
        .agents
        .iter()
        .filter(|r| r.state == AgentState::Working)
        .count();
    let title = format!(" agents · {working} working ");
    let panel = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false });
    frame.render_widget(panel, area);
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
    use crate::app::QueuedInput;
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
    fn queue_height_is_zero_when_empty() {
        // The load-bearing invariant behind "an empty queue renders an
        // identical frame": the layout constraint takes NO space at all, not
        // just a small one.
        assert_eq!(queue_height(0, 24), 0);
        assert_eq!(queue_height(0, 6), 0);
    }

    #[test]
    fn empty_queue_renders_no_queue_box() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("qwen-235b");
        app.history.push(Cell::User("hello world".into()));
        app.running = true;
        app.composer.insert_str("draft");
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        // Same assertions as `renders_transcript_status_and_composer` — an
        // empty queue must not perturb anything already rendered.
        assert!(text.contains("hello world"));
        assert!(text.contains("draft"));
        assert!(
            !text.contains("queued ("),
            "no queue box without queued messages:\n{text}"
        );
    }

    #[test]
    fn queued_messages_render_above_the_composer() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("qwen-235b");
        app.running = true;
        app.queued.push_back("first queued message".into());
        app.queued.push_back("second queued message".into());
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("queued (2)"), "queue count missing:\n{text}");
        assert!(
            text.contains("first queued message"),
            "oldest queued entry missing:\n{text}"
        );
        assert!(
            text.contains("second queued message"),
            "newest queued entry missing:\n{text}"
        );
        assert!(
            text.contains("edit newest"),
            "Up/Esc affordance missing:\n{text}"
        );
    }

    #[test]
    fn queue_box_previews_display_not_the_wire_payload() {
        // The whole point of the display/wire split: the queue box must show
        // the user's clean text, never the internal directive riding along
        // in the wire payload (e.g. Plan mode's read-only prefix). A fixture
        // built with `.into()` (display == wire) can't catch a regression
        // that swaps which field the preview reads — this one can.
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.running = true;
        app.queued.push_back(QueuedInput {
            display: "check the parser".into(),
            wire: "[PLAN MODE — READ-ONLY]\n\ncheck the parser".into(),
        });
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("check the parser"));
        assert!(
            !text.contains("PLAN MODE"),
            "wire directive leaked into the queue box:\n{text}"
        );
    }

    #[test]
    fn transcript_past_u16_rows_saturates_instead_of_wrapping_to_top() {
        // Audit 2026-06-09: `line_count as u16` WRAPS modulo 65,536 — past
        // 65,535 visual rows, `total` collapsed and follow mode pinned the view
        // near the TOP of the transcript. Saturating keeps the newest
        // u16-reachable window visible instead.
        let backend = TestBackend::new(40, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        // 35,000 one-line notices + their separators = 70,000 visual rows.
        for i in 0..35_000 {
            app.history.push(Cell::Notice(format!("line {i}")));
        }
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        // Wrapping cast: 70,000 % 65,536 = 4,464 → the view showed cells near
        // index ~2,230. Saturated: offset 65,527 → cells near index ~32,765.
        assert!(
            !text.contains("line 22"),
            "view must not wrap back to the top of the transcript:\n{text}"
        );
        assert!(
            text.contains("line 3276"),
            "view must pin to the newest reachable window:\n{text}"
        );
    }

    #[test]
    fn mode_picker_modal_lists_all_modes() {
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.modal = Some(Modal::ModePicker { sel: 1 });
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        for label in ["normal", "plan", "YOLO"] {
            assert!(text.contains(label), "missing {label}:\n{text}");
        }
        assert!(
            text.contains("read-only"),
            "plan description shown:\n{text}"
        );
    }

    #[test]
    fn splash_overlay_replaces_everything() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.splash = Some(0);
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("▄▄██▄▄"), "heart art visible:\n{text}");
        assert!(text.contains(env!("CARGO_PKG_VERSION")), "{text}");
        assert!(
            !text.contains("Type a message"),
            "composer hidden during splash:\n{text}"
        );
    }

    #[test]
    fn splash_hides_modals_and_clears_after() {
        use crate::app::KeyEntryModal;
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.modal = Some(Modal::KeyEntry(KeyEntryModal::default()));
        app.splash = Some(0);
        terminal.draw(|f| view(f, &app)).unwrap();
        let during = buffer_text(terminal.backend().buffer());
        assert!(
            !during.contains("OpenRouter API key"),
            "modal hidden under splash:\n{during}"
        );
        app.splash = None;
        terminal.draw(|f| view(f, &app)).unwrap();
        let after = buffer_text(terminal.backend().buffer());
        assert!(
            after.contains("OpenRouter API key"),
            "modal appears at dissolution:\n{after}"
        );
        assert!(!after.contains("▄▄██▄▄"), "art gone:\n{after}");
    }

    #[test]
    fn splash_small_terminal_falls_back_to_one_liner() {
        let backend = TestBackend::new(40, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.splash = Some(0);
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("♥ heartbit"), "{text}");
        assert!(
            !text.contains("▄▄██▄▄"),
            "no art on tiny terminals:\n{text}"
        );
    }

    // A fresh session shows a small identity header (campaign frame evidence:
    // the app opened as a bare wall of notices — no name, no orientation).
    #[test]
    fn empty_transcript_shows_welcome_header() {
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = App::new("m");
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("heartbit"), "product name missing:\n{text}");
        assert!(text.contains("/help"), "help hint missing:\n{text}");
        // …and it disappears once the conversation starts.
        let mut app = App::new("m");
        app.history.push(Cell::User("hi".into()));
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            !text.contains("/help · Shift+Tab"),
            "welcome must clear after first message:\n{text}"
        );
    }

    // While un-pinned (user scrolled up), streaming continues out of sight —
    // the status line must say so, and how to get back (UX: no silent drift).
    #[test]
    fn scrolled_up_shows_a_follow_hint_in_the_status_line() {
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        for i in 0..60 {
            app.history.push(Cell::Agent(format!("line {i}")));
        }
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            !text.contains("scrolled"),
            "pinned view has no hint:\n{text}"
        );
        app.update(crate::msg::Msg::WheelUp);
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            text.contains("scrolled"),
            "un-pinned view must show the scrolled hint:\n{text}"
        );
    }

    // Campaign round-1 frame evidence: the roster panel ate a third of the
    // width on a one-word CHAT turn — every submit seeds the squad as Idle,
    // and the panel showed for idle-only rosters. Display must wait until a
    // sub-agent is actually dispatched (any non-Idle row).
    #[test]
    fn idle_only_roster_does_not_open_the_panel() {
        use crate::app::{AgentRow, AgentState};
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.running = true;
        app.agents = vec![
            AgentRow {
                name: "worker".into(),
                state: AgentState::Idle,
                activity: "available".into(),
                tokens: 0,
            },
            AgentRow {
                name: "researcher".into(),
                state: AgentState::Idle,
                activity: "available".into(),
                tokens: 0,
            },
        ];
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            !text.contains("researcher"),
            "idle-only roster must stay hidden (pure-chat turns keep full width):\n{text}"
        );
        // The moment one agent actually works, the panel appears.
        app.agents[0].state = AgentState::Working;
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            text.contains("worker"),
            "dispatched roster must show:\n{text}"
        );
    }

    // User bug: typing past the right edge of the prompt box became invisible
    // — no wrap, logical-line height, logical-column cursor. The composer must
    // wrap char-exactly, grow a row when the line fills, and keep the cursor
    // inside the box.
    #[test]
    fn composer_wraps_long_input_and_grows() {
        let backend = TestBackend::new(40, 16); // inner composer width = 38
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.composer.insert_str(&"x".repeat(50)); // 38 + 12 → two rows
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            text.contains(&"x".repeat(38)),
            "first wrapped row missing:\n{text}"
        );
        assert!(
            text.contains(&format!("{} ", "x".repeat(12))),
            "wrapped tail row missing — input past the edge is invisible:\n{text}"
        );
        // The cursor must sit INSIDE the frame, after the last typed char.
        // 2 wrapped rows → comp_h = 4 → box spans y 12..16; inner rows are
        // y=13 (first) and y=14 (second, where the cursor lands at col 12).
        let pos = terminal.get_cursor_position().unwrap();
        assert_eq!(
            (pos.x, pos.y),
            (1 + 12, 14),
            "cursor must track the wrapped position:\n{text}"
        );
    }

    #[test]
    fn composer_scrolls_when_input_exceeds_max_height() {
        let backend = TestBackend::new(40, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        // 10 wrapped rows (38 chars each) — far beyond the 6-row inner cap.
        app.composer.insert_str(&"y".repeat(38 * 9 + 5));
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        // The newest row (the tail with the cursor) must be visible…
        assert!(
            text.contains(&format!("{} ", "y".repeat(5))),
            "tail row must stay visible when the composer scrolls:\n{text}"
        );
        // …and the cursor must stay inside the terminal area.
        let pos = terminal.get_cursor_position().unwrap();
        assert!(pos.y < 16, "cursor escaped the area: {pos:?}");
    }

    #[test]
    fn streaming_renders_completed_blocks_live_and_tail_raw() {
        fn joined(lines: &[Line<'static>]) -> String {
            lines
                .iter()
                .flat_map(|l| l.spans.iter())
                .map(|s| s.content.as_ref())
                .collect()
        }
        // Streaming `active`: COMPLETED blocks render styled immediately;
        // the in-flight tail stays raw (no transient half-parsed markup).
        let mut app = App::new("m");
        app.active = Some("# Done\n\nstill **typing".into());
        let t = joined(&transcript_lines(&app));
        assert!(
            !t.contains('#'),
            "completed heading must render styled while streaming: {t}"
        );
        assert!(
            t.contains("**typing"),
            "the in-flight tail must stay raw: {t}"
        );
        // Finalized agent cell: full Markdown rendered (markers gone).
        let mut app2 = App::new("m");
        app2.history.push(Cell::Agent("# Heading **bold**".into()));
        let t = joined(&transcript_lines(&app2));
        assert!(
            !t.contains('#') && !t.contains('*'),
            "finalized cell must render markdown: {t}"
        );
    }

    #[test]
    fn live_reasoning_renders_dimmed_above_streaming_answer() {
        let mut app = App::new("m");
        app.active_reasoning = Some("step one".into());
        app.active = Some("partial answer".into());
        let lines = transcript_lines(&app);
        let joined: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        assert!(joined.contains("thinking"), "live reasoning shows a header");
        assert!(joined.contains("step one"), "live reasoning body shows");
        // Reasoning must come before the answer in the flattened lines.
        let r_pos = joined.find("step one").unwrap();
        let a_pos = joined.find("partial answer").unwrap();
        assert!(r_pos < a_pos, "reasoning renders above the answer");
    }

    #[test]
    fn roster_and_todos_both_visible_during_a_multi_agent_run() {
        use crate::app::{AgentRow, AgentState, TodoRow, TodoStatus};
        // The reported bug: the task list only appeared AFTER stopping, because the
        // roster owned the whole side column while running. They must now stack.
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.multi_agent = true;
        app.running = true;
        app.agents = vec![AgentRow {
            name: "worker".into(),
            state: AgentState::Working,
            activity: "editing".into(),
            tokens: 0,
        }];
        app.todos = vec![TodoRow {
            content: "PLAN_ITEM_ONE".into(),
            status: TodoStatus::InProgress,
        }];
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            text.contains("worker"),
            "roster must show during the run:\n{text}"
        );
        assert!(
            text.contains("PLAN_ITEM_ONE"),
            "the task list must ALSO be visible during the run, not only after stopping:\n{text}"
        );
    }

    #[test]
    fn newest_content_visible_even_with_roster_panel_narrowing_transcript() {
        use crate::app::{AgentRow, AgentState};
        // The roster panel steals width; the scroll offset must be computed from
        // the NARROWED transcript width (post-split), or the newest line is clipped.
        let backend = TestBackend::new(80, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.multi_agent = true;
        app.running = true;
        app.agents = vec![AgentRow {
            name: "worker".into(),
            state: AgentState::Working,
            activity: "x".into(),
            tokens: 0,
        }];
        for i in 0..8 {
            app.history.push(Cell::Agent(format!(
                "filler {i} a long assistant line that wraps across several visual rows in the narrowed transcript column for sure"
            )));
        }
        app.history.push(Cell::Agent("NEWEST_MARKER".into()));
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            text.contains("NEWEST_MARKER"),
            "newest content clipped when the roster panel narrows the transcript:\n{text}"
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
    fn question_modal_renders_header_options_descriptions() {
        use crate::app::{Modal, QuestionModal};
        use heartbit_core::tool::builtins::{Question, QuestionOption, QuestionRequest};
        let backend = TestBackend::new(90, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.modal = Some(Modal::Question(QuestionModal {
            request: QuestionRequest {
                questions: vec![Question {
                    question: "Which storage backend?".into(),
                    header: "Storage".into(),
                    options: vec![
                        QuestionOption {
                            label: "sqlite".into(),
                            description: "single file, zero ops".into(),
                        },
                        QuestionOption {
                            label: "postgres".into(),
                            description: "full server".into(),
                        },
                    ],
                    multiple: false,
                }],
            },
            reply: None,
            current: 0,
            selected: 0,
            picked: vec![false; 2],
            answers: Vec::new(),
        }));
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("the agent asks"), "title:\n{text}");
        assert!(text.contains("Storage"), "header chip:\n{text}");
        assert!(text.contains("Which storage backend?"), "question:\n{text}");
        assert!(
            text.contains("sqlite") && text.contains("postgres"),
            "options:\n{text}"
        );
        assert!(
            text.contains("single file, zero ops"),
            "descriptions:\n{text}"
        );
    }

    #[test]
    fn model_picker_advisor_target_renders_advisor_title() {
        use crate::app::{Modal, ModelPicker, ModelTarget};
        let backend = TestBackend::new(72, 22);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.modal = Some(Modal::ModelPicker(ModelPicker {
            target: ModelTarget::Advisor,
            ..ModelPicker::default()
        }));
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            text.contains("select advisor model"),
            "advisor title missing:\n{text}"
        );
    }

    #[test]
    fn multi_agent_roster_panel_renders_agents_and_state() {
        use crate::app::{AgentRow, AgentState};
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.multi_agent = true;
        app.running = true; // the live panel shows only while the turn runs
        app.agents = vec![
            AgentRow {
                name: "worker".into(),
                state: AgentState::Working,
                activity: "write".into(),
                tokens: 0,
            },
            AgentRow {
                name: "researcher".into(),
                state: AgentState::Done,
                activity: "done".into(),
                tokens: 1200,
            },
        ];
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("agents"), "roster title missing:\n{text}");
        assert!(
            text.contains("worker") && text.contains("researcher"),
            "agent names missing:\n{text}"
        );
        assert!(text.contains("write"), "live activity missing:\n{text}");
        assert!(text.contains('✓'), "done state icon missing:\n{text}");
    }

    #[test]
    fn roster_panel_hidden_once_the_turn_is_idle() {
        use crate::app::{AgentRow, AgentState};
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.multi_agent = true;
        app.running = false; // turn finished — the live panel must disappear
        app.agents = vec![AgentRow {
            name: "worker".into(),
            state: AgentState::Done,
            activity: "done".into(),
            tokens: 100,
        }];
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            !text.contains("agents ·"),
            "roster panel must hide when no agent is running:\n{text}"
        );
    }

    #[test]
    fn status_bar_wraps_to_two_lines_when_overflowing() {
        // Narrow terminal + long model/advisor pair: the metrics (incl. the
        // run state) must wrap to a second row instead of falling off-screen.
        let backend = TestBackend::new(58, 8);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("mistralai/mistral-medium-3-5");
        app.frontier_model = Some("anthropic/claude-opus-4.6".into());
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("advised by"), "identity row:\n{text}");
        assert!(text.contains("ready"), "state must stay visible:\n{text}");
        let model_row = text
            .lines()
            .find(|l| l.contains("mistral-medium"))
            .expect("model row");
        assert!(
            !model_row.contains("ready"),
            "metrics must be on their own row when narrow:\n{text}"
        );
    }

    #[test]
    fn status_bar_stays_single_line_when_it_fits() {
        let backend = TestBackend::new(140, 8);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.frontier_model = Some("a/opus".into());
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        let status_row = text
            .lines()
            .find(|l| l.contains("advised by"))
            .expect("status row");
        assert!(
            status_row.contains("ready"),
            "wide terminal keeps one status row:\n{text}"
        );
    }

    #[test]
    fn status_line_shows_advised_by_when_advisor_set() {
        let backend = TestBackend::new(110, 6);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("mistralai/mistral-medium");
        app.frontier_model = Some("anthropic/claude-opus-4".into());
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            text.contains("advised by anthropic/claude-opus-4"),
            "advisor model missing from status line:\n{text}"
        );
    }

    #[test]
    fn status_line_omits_advised_by_when_advisor_unset() {
        let backend = TestBackend::new(110, 6);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = App::new("mistralai/mistral-medium");
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            !text.contains("advised by"),
            "no advisor → no 'advised by' segment:\n{text}"
        );
    }

    #[test]
    fn status_line_shows_effort_when_set() {
        use crate::app::EffortLevel;
        let backend = TestBackend::new(110, 6);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("mistralai/mistral-medium");
        app.effort = EffortLevel::High;
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            text.contains("effort:high"),
            "effort level missing from status line:\n{text}"
        );
    }

    #[test]
    fn status_line_omits_effort_when_off() {
        // Off is the default — the status line must render bit-for-bit like
        // it did before this feature existed.
        let backend = TestBackend::new(110, 6);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = App::new("mistralai/mistral-medium");
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            !text.contains("effort:"),
            "Off must not show an effort segment:\n{text}"
        );
    }

    #[test]
    fn effort_picker_renders_all_levels_and_marks_the_current_one() {
        use crate::app::{EffortLevel, Modal};
        let backend = TestBackend::new(80, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.effort = EffortLevel::Medium;
        app.modal = Some(Modal::EffortPicker { sel: 0 });
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        for level in EffortLevel::ALL {
            assert!(
                text.contains(level.label()),
                "{} missing from the picker:\n{text}",
                level.label()
            );
        }
        assert!(
            text.contains("reasoning effort"),
            "picker title missing:\n{text}"
        );
    }

    #[test]
    fn status_line_shows_context_bar_when_limit_known() {
        use crate::models::ModelEntry;
        let backend = TestBackend::new(100, 6);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("qwen/q");
        app.models = vec![ModelEntry {
            id: "qwen/q".into(),
            name: "Q".into(),
            context: Some(10_000),
        }];
        app.context_tokens = 5_000; // 50%
        app.last_ttft_ms = 1400;
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("50% ctx"), "context % missing:\n{text}");
        assert!(text.contains('█'), "context gauge missing:\n{text}");
        assert!(text.contains("1.4s ttft"), "ttft missing:\n{text}");
    }

    #[test]
    fn status_line_falls_back_to_raw_ctx_without_catalog() {
        let backend = TestBackend::new(100, 6);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("unknown/model"); // not in (empty) catalog
        app.context_tokens = 12_800;
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            text.contains("12.8k ctx"),
            "raw ctx fallback missing:\n{text}"
        );
        assert!(
            !text.contains('%'),
            "no percent without a known limit:\n{text}"
        );
    }

    #[test]
    fn status_line_shows_permission_mode_when_not_default() {
        use crate::app::PermissionMode;
        let backend = TestBackend::new(100, 6);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        // Normal → not shown (the app now DEFAULTS to YOLO; set Normal explicitly)
        app.permission_mode = PermissionMode::Normal;
        terminal.draw(|f| view(f, &app)).unwrap();
        assert!(!buffer_text(terminal.backend().buffer()).contains("YOLO"));
        // YOLO → shown
        app.permission_mode = PermissionMode::Yolo;
        terminal.draw(|f| view(f, &app)).unwrap();
        assert!(
            buffer_text(terminal.backend().buffer()).contains("YOLO"),
            "execution mode should show in the status line"
        );
    }

    #[test]
    fn human_helpers_format_compactly() {
        assert_eq!(human_tokens(999), "999");
        assert_eq!(human_tokens(1500), "1.5k");
        assert_eq!(human_ms(420), "420ms");
        assert_eq!(human_ms(1400), "1.4s");
    }

    #[test]
    fn todo_panel_renders_tasks_and_progress() {
        use crate::app::{TodoRow, TodoStatus};
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.running = true;
        app.todos = vec![
            TodoRow {
                content: "implement diffs".into(),
                status: TodoStatus::Completed,
            },
            TodoRow {
                content: "todo panel".into(),
                status: TodoStatus::InProgress,
            },
        ];
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("tasks"), "todo title missing:\n{text}");
        assert!(text.contains("1/2 done"), "progress count missing:\n{text}");
        assert!(
            text.contains("implement diffs") && text.contains("todo panel"),
            "task content missing:\n{text}"
        );
    }

    #[test]
    fn roster_and_todos_stack_in_the_panel_column_while_running() {
        use crate::app::{AgentRow, AgentState, TodoRow, TodoStatus};
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.multi_agent = true;
        app.running = true;
        app.agents = vec![AgentRow {
            name: "worker".into(),
            state: AgentState::Working,
            activity: "write".into(),
            tokens: 0,
        }];
        app.todos = vec![TodoRow {
            content: "a task".into(),
            status: TodoStatus::Pending,
        }];
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        // Both panels share the column (stacked) — the task list is no longer
        // hidden by the roster during the run.
        assert!(
            text.contains("agents"),
            "roster shows while running:\n{text}"
        );
        assert!(
            text.contains("tasks"),
            "the task list must ALSO show, stacked under the roster:\n{text}"
        );
    }

    #[test]
    fn no_roster_panel_in_single_agent_mode() {
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = App::new("m"); // multi_agent = false
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(
            !text.contains("agents ·"),
            "no roster panel when single-agent:\n{text}"
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
        // full input shown (no 48-char blind truncation)
        assert!(
            text.contains("rm -rf /tmp/x"),
            "full command missing:\n{text}"
        );
    }

    #[test]
    fn approval_modal_shows_a_diff_for_an_edit() {
        use crate::msg::PendingTool;
        use std::sync::mpsc::sync_channel;
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        let (tx, _rx) = sync_channel(1);
        app.update(crate::msg::Msg::Approval {
            tools: vec![PendingTool {
                name: "edit".into(),
                input: r#"{"file_path":"f.rs","old_string":"old code","new_string":"new code"}"#
                    .into(),
            }],
            reply: tx,
        });
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("-old code"), "diff removal missing:\n{text}");
        assert!(text.contains("+new code"), "diff addition missing:\n{text}");
    }

    #[test]
    fn approval_modal_hint_lists_every_answer_key() {
        use crate::msg::PendingTool;
        use std::sync::mpsc::sync_channel;
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        let (tx, _rx) = sync_channel(1);
        app.update(crate::msg::Msg::Approval {
            tools: vec![PendingTool {
                name: "bash".into(),
                input: "{}".into(),
            }],
            reply: tx,
        });
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        // Every key `handle_approval_key` actually handles must be advertised.
        for k in ["y", "n", "a", "d"] {
            assert!(
                text.contains(&format!("[{k}]")),
                "approval hint must advertise [{k}]:\n{text}"
            );
        }
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
