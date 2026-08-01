//! Render Markdown agent output into styled ratatui [`Line`]s. Headings, bold,
//! italic, inline code, fenced code blocks, and lists are styled; everything
//! else degrades to readable text. Pure (no I/O) so it is unit-testable.
//!
//! Fenced code blocks are additionally syntax-highlighted via `syntect`. Its
//! syntax/theme data are embedded bincode dumps — `SyntaxSet::load_defaults_newlines()`
//! and `ThemeSet::load_defaults()` never touch disk — so this module stays I/O-free.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

/// The colour used for inline code, and for fenced code blocks whose language
/// is unknown/unset (syntax highlighting falls back to this flat colour).
const CODE: Color = Color::Yellow;
/// The colour used for headings.
const HEADING: Color = Color::Cyan;

/// A bundled syntect theme name guaranteed present in `ThemeSet::load_defaults()`
/// (asserted by syntect's own test suite) — used when no `syntax_theme` is
/// configured, or the configured name doesn't resolve.
const DEFAULT_THEME_NAME: &str = "base16-ocean.dark";

fn syntax_set() -> &'static SyntaxSet {
    static SET: OnceLock<SyntaxSet> = OnceLock::new();
    SET.get_or_init(SyntaxSet::load_defaults_newlines)
}

fn theme_set() -> &'static ThemeSet {
    static SET: OnceLock<ThemeSet> = OnceLock::new();
    SET.get_or_init(ThemeSet::load_defaults)
}

/// Resolve `name` to a bundled theme: the named theme if it exists, else
/// [`DEFAULT_THEME_NAME`], else (unreachable in practice — syntect always
/// bundles it — but avoids a panic) `Theme::default()`.
fn resolve_theme(name: Option<&str>) -> Theme {
    let set = theme_set();
    name.and_then(|n| set.themes.get(n))
        .or_else(|| set.themes.get(DEFAULT_THEME_NAME))
        .cloned()
        .unwrap_or_default()
}

/// The theme `render()`/`render_streaming()` use — resolved once, never
/// overridden (a per-session choice lives on [`MarkdownCache`] instead, since
/// that's the type main.rs actually constructs from `tui.toml`).
fn default_theme() -> &'static Theme {
    static THEME: OnceLock<Theme> = OnceLock::new();
    THEME.get_or_init(|| resolve_theme(None))
}

/// The first whitespace/comma-delimited token of a fence info string —
/// "rust,ignore" and "rust title=x" both mean "rust".
fn language_token(info: &str) -> &str {
    info.trim().split([' ', ',', '\t']).next().unwrap_or("")
}

fn syntect_style_to_ratatui(style: syntect::highlighting::Style) -> Style {
    let c = style.foreground;
    Style::default().fg(Color::Rgb(c.r, c.g, c.b))
}

/// Render one fenced/indented code block's buffered text into styled lines,
/// syntax-highlighted via `theme` when `lang` resolves to a known syntax,
/// otherwise kept in the flat [`CODE`] colour (unknown/absent language).
///
/// `buf` is the raw text pulldown-cmark handed us, which carries the source's
/// own trailing `\n` (the line before the closing fence). We strip exactly
/// that one trailing newline before splitting into lines, and TRIM (not just
/// exact-match) a trailing `\n` off every highlighted range's text — both
/// paths then emit exactly one [`Line`] per source line, with identical
/// characters, which is the invariant
/// `highlighting_preserves_code_characters_and_line_count` checks.
///
/// The trim (rather than an exact `*t == "\n"` match) matters: syntect merges
/// the line terminator into the PRECEDING range whenever no scope changes at
/// end-of-line — e.g. a `//` comment or an in-progress multi-line string —
/// so a range's text can be `" trailing comment\n"`, not a separate `"\n"`
/// range. An exact match misses that and leaves a literal newline character
/// embedded inside a `Span`, silently breaking the invariant for any block
/// containing a comment or multi-line string that isn't the block's last
/// line (a comment or string AS the last line is already clean, since the
/// block's own trailing newline was stripped above).
fn highlight_code_block(buf: &str, lang: Option<&str>, theme: &Theme) -> Vec<Line<'static>> {
    let text = buf.strip_suffix('\n').unwrap_or(buf);
    if text.is_empty() {
        return Vec::new();
    }
    match lang.and_then(|l| syntax_set().find_syntax_by_token(l)) {
        Some(syntax) => {
            let mut h = HighlightLines::new(syntax, theme);
            LinesWithEndings::from(text)
                .map(|line| match h.highlight_line(line, syntax_set()) {
                    Ok(ranges) => Line::from(
                        ranges
                            .into_iter()
                            .filter_map(|(style, t)| {
                                let t = t.strip_suffix('\n').unwrap_or(t);
                                (!t.is_empty()).then(|| {
                                    Span::styled(t.to_string(), syntect_style_to_ratatui(style))
                                })
                            })
                            .collect::<Vec<_>>(),
                    ),
                    // A syntect parse error yields zero ranges — falling back
                    // to `unwrap_or_default()` would silently DELETE this
                    // line's characters. Keep them, in the flat CODE colour.
                    Err(_) => {
                        let flat = line.strip_suffix('\n').unwrap_or(line);
                        if flat.is_empty() {
                            Line::default()
                        } else {
                            Line::from(Span::styled(flat.to_string(), Style::default().fg(CODE)))
                        }
                    }
                })
                .collect()
        }
        None => text
            .split('\n')
            .map(|line| Line::from(Span::styled(line.to_string(), Style::default().fg(CODE))))
            .collect(),
    }
}

/// Accumulates Markdown events into styled lines.
struct Renderer<'t> {
    lines: Vec<Line<'static>>,
    spans: Vec<Span<'static>>,
    style: Style,
    stack: Vec<Style>,
    /// Per-level ordered-list counters (`None` = unordered).
    list: Vec<Option<u64>>,
    in_code: bool,
    /// Buffered text of the currently-open code block, accumulated across
    /// (possibly several) `Event::Text`s — syntax highlighting needs the
    /// whole block at once, not line fragments, so nothing is emitted to
    /// `spans`/`lines` until `TagEnd::CodeBlock`.
    code_buf: String,
    /// The open code block's fence language token (`None` for an indented
    /// block or an unlabeled fence).
    code_lang: Option<String>,
    theme: &'t Theme,
}

impl<'t> Renderer<'t> {
    fn new(theme: &'t Theme) -> Self {
        Self {
            lines: Vec::new(),
            spans: Vec::new(),
            style: Style::default(),
            stack: Vec::new(),
            list: Vec::new(),
            in_code: false,
            code_buf: String::new(),
            code_lang: None,
            theme,
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

/// Byte index of the last SAFE block boundary for live streaming: the end of
/// the last blank line that is OUTSIDE any open ``` fence. Everything before
/// it is a completed Markdown block (stable across future deltas); everything
/// after is the in-flight tail. 0 = nothing completed yet.
fn safe_split(text: &str) -> usize {
    let mut in_fence = false;
    let mut split = 0usize;
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
        } else if trimmed.is_empty() && !in_fence && line.ends_with('\n') {
            split = offset + line.len();
        }
        offset += line.len();
    }
    split
}

/// Live-streaming render: completed blocks styled, the in-flight tail raw —
/// partial markers (`**term` before its close) never flash styled, which is
/// what previously forced streaming to render all-plain.
pub fn render_streaming(text: &str) -> Vec<Line<'static>> {
    let split = safe_split(text);
    let mut lines = if split > 0 {
        render(&text[..split])
    } else {
        Vec::new()
    };
    let tail = &text[split..];
    if !tail.is_empty() {
        if !lines.is_empty() {
            lines.push(Line::raw(""));
        }
        lines.extend(tail.split('\n').map(|l| Line::raw(l.to_string())));
    }
    lines
}

/// Render Markdown `text` into styled ratatui lines.
pub fn render(text: &str) -> Vec<Line<'static>> {
    render_with_theme(text, default_theme())
}

fn render_with_theme(text: &str, theme: &Theme) -> Vec<Line<'static>> {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    let mut r = Renderer::new(theme);

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
                Tag::CodeBlock(kind) => {
                    if !r.spans.is_empty() {
                        r.flush();
                    }
                    r.in_code = true;
                    r.code_buf.clear();
                    r.code_lang = match kind {
                        CodeBlockKind::Fenced(info) if !info.trim().is_empty() => {
                            Some(language_token(&info).to_string())
                        }
                        _ => None,
                    };
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
                    r.in_code = false;
                    let buf = std::mem::take(&mut r.code_buf);
                    let lang = r.code_lang.take();
                    let block = highlight_code_block(&buf, lang.as_deref(), r.theme);
                    r.lines.extend(block);
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
                    r.code_buf.push_str(&t);
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

/// Per-frame memoization of [`render`]'s output, keyed by the SOURCE TEXT —
/// never a hash: a collision would silently render a different cell's
/// content, and no test could ever catch it.
///
/// Interior-mutable (`RefCell`, not `&mut`) because `view()` only takes
/// `&App`. The reducer must stay pure — see `App::update`'s `Msg::Resize`
/// arm, which does NOT touch this cache. That's safe (not a missed
/// invalidation): entries hold LOGICAL lines, and terminal-width wrapping
/// happens at draw time via `Paragraph::wrap` against the live
/// `transcript_area.width` (`ui::view`), never against anything cached here —
/// see `Cell::to_lines`'s own doc ("the caller wraps to width"). So there is
/// no width-derived state in this cache to go stale on a resize.
pub(crate) struct MarkdownCache {
    theme: Theme,
    entries: RefCell<HashMap<String, Vec<Line<'static>>>>,
    /// Keys rendered since the last `begin_frame()` call.
    touched: RefCell<HashSet<String>>,
}

impl Default for MarkdownCache {
    fn default() -> Self {
        Self::new(None)
    }
}

impl MarkdownCache {
    /// `theme_name` is the configured `tui.toml` `syntax_theme`, if any — an
    /// absent or unresolvable name falls back to the default theme (see
    /// [`resolve_theme`]).
    pub(crate) fn new(theme_name: Option<&str>) -> Self {
        Self {
            theme: resolve_theme(theme_name),
            entries: RefCell::new(HashMap::new()),
            touched: RefCell::new(HashSet::new()),
        }
    }

    /// Cached [`render`]: a hit returns memoized lines, a miss renders (with
    /// this cache's theme), stores, and returns them. Marks `src` touched for
    /// this frame's sweep either way.
    pub(crate) fn render(&self, src: &str) -> Vec<Line<'static>> {
        self.touched.borrow_mut().insert(src.to_string());
        if let Some(hit) = self.entries.borrow().get(src) {
            return hit.clone();
        }
        let out = render_with_theme(src, &self.theme);
        self.entries
            .borrow_mut()
            .insert(src.to_string(), out.clone());
        out
    }

    /// Sweep entries not touched since the previous `begin_frame()`, then
    /// reset the touched set for the new frame. Called exactly once, as the
    /// first statement of `ui::transcript_lines`, so the sweep is
    /// deterministic: a cell that scrolled out of the transcript (and so was
    /// never re-rendered) is evicted one frame later instead of growing
    /// forever.
    pub(crate) fn begin_frame(&self) {
        let touched = self.touched.borrow();
        self.entries.borrow_mut().retain(|k, _| touched.contains(k));
        drop(touched);
        self.touched.borrow_mut().clear();
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.borrow().len()
    }
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

    // Live markdown while streaming: COMPLETED blocks (delimited by a blank
    // line outside code fences) render styled immediately; only the in-flight
    // tail stays raw — so partial markers (`**term` before its close) never
    // flash styled, which is why streaming was previously all-plain.
    #[test]
    fn streaming_styles_completed_blocks_and_keeps_tail_raw() {
        let lines = render_streaming("# Done Title\n\nnow **partial");
        // The completed heading is styled (hashes dropped, coloured)…
        let s = span_containing(&lines, "Done Title").expect("heading text");
        assert_eq!(s.style.fg, Some(HEADING));
        assert!(!all_text(&lines).contains('#'));
        // …while the unfinished tail keeps its raw markers untouched.
        assert!(
            all_text(&lines).contains("**partial"),
            "tail must stay raw: {}",
            all_text(&lines)
        );
    }

    #[test]
    fn streaming_with_no_completed_block_is_all_raw() {
        let lines = render_streaming("just **typing");
        assert_eq!(all_text(&lines).trim(), "just **typing");
    }

    #[test]
    fn streaming_open_fence_holds_the_whole_fence_raw() {
        // A blank line INSIDE an open ``` fence is not a block boundary — the
        // fence stays raw until it closes (else half a code block would
        // render as styled prose).
        let text = "intro\n\n```rust\nlet a = 1;\n\nlet b = 2;";
        let lines = render_streaming(text);
        let flat = all_text(&lines);
        assert!(flat.contains("```rust"), "open fence stays raw: {flat}");
        assert!(flat.contains("let b = 2;"));
        // The completed intro before the fence is processed as markdown
        // (passthrough text here).
        assert!(flat.contains("intro"));
    }

    #[test]
    fn streaming_closed_fence_renders_styled() {
        let text = "```rust\nlet a = 1;\n```\n\ntail **wip";
        let lines = render_streaming(text);
        let flat = all_text(&lines);
        // Fence markers are dropped by the renderer once the block completes…
        assert!(!flat.contains("```"), "closed fence renders styled: {flat}");
        assert!(flat.contains("let a = 1;"));
        // …and the tail stays raw.
        assert!(flat.contains("**wip"));
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
        // Untagged deliberately (maintainer ruling, task 8): a LANGUAGE-TAGGED
        // fence is now syntax-highlighted by design (multiple colours — see
        // `fenced_rust_block_is_syntax_highlighted`), so it no longer renders
        // as one flat-CODE-coloured span. This test's purpose was always the
        // untagged/flat-colour path — do not restore the `rust` tag, it will
        // re-break this assertion.
        let lines = render("```\nfn main() {}\n```");
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
    fn fenced_rust_block_is_syntax_highlighted() {
        let lines = render("```rust\nfn main() {}\n```");
        let colours: std::collections::HashSet<_> = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .filter_map(|s| s.style.fg)
            .collect();
        assert!(
            colours.len() > 1,
            "a highlighted block uses more than one colour"
        );
    }

    #[test]
    fn unknown_language_falls_back_to_flat_code_colour() {
        let lines = render("```notalanguage\nfn main() {}\n```");
        let colours: std::collections::HashSet<_> = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .filter_map(|s| s.style.fg)
            .collect();
        assert_eq!(
            colours.len(),
            1,
            "unknown languages keep the flat code colour"
        );
    }

    #[test]
    fn highlighting_preserves_code_characters_and_line_count() {
        // THE invariant: highlighting must not change one character or one line,
        // or every existing markdown assertion silently becomes wrong.
        //
        // The fixture MUST include a comment line followed by another line,
        // and a multi-line string: syntect merges the line terminator into
        // the PRECEDING range whenever no scope changes at end-of-line (a
        // `//` comment or an in-progress string are exactly that), so a
        // fixture with no such line cannot catch a filter that only drops
        // ranges whose text is EXACTLY "\n" — a comment as the block's LAST
        // line would be clean too (the block's own trailing newline is
        // stripped before highlighting), so the comment must NOT be last.
        let src = "```rust\nfn main() {\n    // trailing comment\n    let s = \"line one\nline two\";\n}\n```";
        let hl = render(src);
        let flat = render(&src.replace("```rust", "```"));
        assert_eq!(all_text(&hl), all_text(&flat));
        assert_eq!(hl.len(), flat.len());
    }

    #[test]
    fn fence_info_string_takes_the_first_token() {
        // "```rust,ignore" and "```rust title=x" both mean rust.
        for info in ["rust", "rust,ignore", "rust title=x"] {
            assert_eq!(language_token(info), "rust");
        }
    }

    #[test]
    fn markdown_cache_hit_matches_the_uncached_render() {
        let cache = MarkdownCache::default();
        let src = "```rust\nfn main() {}\n```";
        cache.begin_frame();
        let first = cache.render(src);
        let second = cache.render(src); // served from the cache
        assert_eq!(first, second);
        assert_eq!(first, render(src));
    }

    #[test]
    fn markdown_cache_sweeps_entries_not_reused_next_frame() {
        let cache = MarkdownCache::default();
        cache.begin_frame();
        let _ = cache.render("```rust\nfn a() {}\n```");
        assert_eq!(cache.len(), 1);
        cache.begin_frame(); // new frame, entry not touched
        cache.begin_frame(); // …and swept
        assert_eq!(cache.len(), 0, "unused entries must not grow forever");
    }

    #[test]
    fn cache_honours_a_configured_theme_and_falls_back_on_an_unknown_name() {
        // The wiring for tui.toml's `syntax_theme`: a resolvable name must
        // actually change the highlighted colours (proving it reached
        // syntect), and an unknown name must fall back to the same output as
        // the default-constructed cache rather than panicking or going flat.
        let src = "```rust\nfn main() {}\n```";
        let default_out = MarkdownCache::default().render(src);
        let other = MarkdownCache::new(Some("InspiredGitHub")).render(src);
        assert_ne!(
            default_out, other,
            "a different bundled theme must change the rendered colours"
        );
        let unknown = MarkdownCache::new(Some("not-a-real-theme")).render(src);
        assert_eq!(
            unknown, default_out,
            "an unresolvable theme name falls back to the default theme"
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
