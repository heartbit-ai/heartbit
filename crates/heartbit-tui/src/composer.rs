//! Multiline input composer — a small text editor for the prompt box. Pure: no
//! terminal I/O, so it is fully unit-testable. Lines are stored as `Vec<char>`
//! to make cursor math char-correct (no byte/UTF-8 boundary bugs).

/// Greedy display-cell packing of one logical line into rows of at most
/// `width` cells: returns char-index ranges. A wide char never splits across
/// a boundary; an exactly-full final row yields a trailing empty range (the
/// fresh-input row the cursor lands on). Zero-width chars pack freely.
fn pack_line(line: &[char], width: usize) -> Vec<(usize, usize)> {
    use unicode_width::UnicodeWidthChar;
    let mut rows = Vec::new();
    let mut start = 0usize;
    let mut used = 0usize;
    for (i, ch) in line.iter().enumerate() {
        let w = ch.width().unwrap_or(0);
        if used + w > width && used > 0 {
            rows.push((start, i));
            start = i;
            used = 0;
        }
        used += w;
    }
    rows.push((start, line.len()));
    if used >= width {
        rows.push((line.len(), line.len()));
    }
    rows
}

/// A multiline editor with a cursor and submit-history recall.
#[derive(Debug)]
pub struct Composer {
    lines: Vec<Vec<char>>,
    row: usize,
    col: usize,
    history: Vec<String>,
    /// `None` = editing the live buffer; `Some(i)` = viewing `history[i]`.
    hist_pos: Option<usize>,
}

impl Default for Composer {
    fn default() -> Self {
        Self::new()
    }
}

impl Composer {
    pub fn new() -> Self {
        Self {
            lines: vec![Vec::new()],
            row: 0,
            col: 0,
            history: Vec::new(),
            hist_pos: None,
        }
    }

    /// The current buffer text (lines joined with `\n`).
    pub fn text(&self) -> String {
        self.lines
            .iter()
            .map(|l| l.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// True when the buffer has no characters at all.
    pub fn is_empty(&self) -> bool {
        self.lines.iter().all(|l| l.is_empty())
    }

    /// The logical (unwrapped) lines — test-only since the renderer moved to
    /// the wrap-aware [`Self::wrap_lines`].
    #[cfg(test)]
    pub fn render_lines(&self) -> Vec<String> {
        self.lines.iter().map(|l| l.iter().collect()).collect()
    }

    /// (row, col) LOGICAL cursor position in chars — test-only since the
    /// renderer moved to the wrap-aware [`Self::visual_cursor`].
    #[cfg(test)]
    pub fn cursor(&self) -> (usize, usize) {
        (self.row, self.col)
    }

    /// Display-cell wrap of the buffer into visual rows of at most `width`
    /// CELLS (wide CJK/emoji chars take 2) — the SAME math `visual_cursor`
    /// uses, so the rendered text and the cursor can never disagree. A wide
    /// char never splits across rows; an exactly-full row grows a fresh,
    /// empty row (where continued typing lands).
    pub fn wrap_lines(&self, width: usize) -> Vec<String> {
        let width = width.max(1);
        let mut rows = Vec::new();
        for line in &self.lines {
            for (s, e) in pack_line(line, width) {
                rows.push(line[s..e].iter().collect());
            }
        }
        rows
    }

    /// Cursor position in WRAPPED rows for a given width — (visual row,
    /// display-cell column), same packing rule as [`Self::wrap_lines`].
    pub fn visual_cursor(&self, width: usize) -> (usize, usize) {
        use unicode_width::UnicodeWidthChar;
        let width = width.max(1);
        let rows_above: usize = self.lines[..self.row]
            .iter()
            .map(|l| pack_line(l, width).len())
            .sum();
        let line = &self.lines[self.row];
        let packs = pack_line(line, width);
        let mut row_in_line = packs.len() - 1;
        for (r, &(_, e)) in packs.iter().enumerate() {
            if self.col < e || (self.col == e && r == packs.len() - 1) {
                row_in_line = r;
                break;
            }
        }
        let (start, _) = packs[row_in_line];
        let cell: usize = line[start..self.col]
            .iter()
            .map(|c| c.width().unwrap_or(0))
            .sum();
        (rows_above + row_in_line, cell)
    }

    /// The submit-history entries (oldest→newest) — for Ctrl+R reverse search.
    pub fn history(&self) -> &[String] {
        &self.history
    }

    /// The `@token` being typed immediately before the cursor on the current
    /// line (without the `@`), for file-mention autocomplete — or `None`.
    pub fn mention_prefix(&self) -> Option<String> {
        let line = &self.lines[self.row];
        let before: String = line[..self.col].iter().collect();
        let tok = before.rsplit(char::is_whitespace).next().unwrap_or("");
        tok.strip_prefix('@').map(|s| s.to_string())
    }

    /// Replace the `@token` before the cursor with `path` + a trailing space.
    pub fn complete_mention(&mut self, path: &str) {
        let line = &self.lines[self.row];
        let before: String = line[..self.col].iter().collect();
        let after: Vec<char> = line[self.col..].to_vec();
        // strip the trailing `@token`
        let cut = before.rfind('@').unwrap_or(before.len());
        let mut head: Vec<char> = before[..cut].chars().collect();
        head.extend(format!("{path} ").chars());
        let new_col = head.len();
        head.extend(after);
        self.lines[self.row] = head;
        self.col = new_col;
        self.hist_pos = None;
    }

    pub fn insert_char(&mut self, c: char) {
        self.hist_pos = None;
        self.lines[self.row].insert(self.col, c);
        self.col += 1;
    }

    /// Insert a hard newline at the cursor (Shift+Enter), splitting the line.
    pub fn newline(&mut self) {
        self.hist_pos = None;
        let tail = self.lines[self.row].split_off(self.col);
        self.lines.insert(self.row + 1, tail);
        self.row += 1;
        self.col = 0;
    }

    /// Insert a string (e.g. a bracketed paste), honoring embedded newlines.
    pub fn insert_str(&mut self, s: &str) {
        for c in s.chars() {
            if c == '\n' {
                self.newline();
            } else if c != '\r' {
                self.insert_char(c);
            }
        }
    }

    pub fn backspace(&mut self) {
        self.hist_pos = None;
        if self.col > 0 {
            self.lines[self.row].remove(self.col - 1);
            self.col -= 1;
        } else if self.row > 0 {
            // Join with the previous line.
            let cur = self.lines.remove(self.row);
            self.row -= 1;
            self.col = self.lines[self.row].len();
            self.lines[self.row].extend(cur);
        }
    }

    pub fn move_left(&mut self) {
        if self.col > 0 {
            self.col -= 1;
        } else if self.row > 0 {
            self.row -= 1;
            self.col = self.lines[self.row].len();
        }
    }

    pub fn move_right(&mut self) {
        if self.col < self.lines[self.row].len() {
            self.col += 1;
        } else if self.row + 1 < self.lines.len() {
            self.row += 1;
            self.col = 0;
        }
    }

    /// Clear the live buffer WITHOUT recording it in history (used for slash
    /// commands so a `/key <token>` secret is never recalled via the Up arrow).
    pub fn clear(&mut self) {
        self.lines = vec![Vec::new()];
        self.row = 0;
        self.col = 0;
        self.hist_pos = None;
    }

    /// Replace the buffer with `s` (cursor at end), without recording history.
    /// Used by slash-command autocompletion.
    pub fn set_text(&mut self, s: &str) {
        self.load(s);
        self.hist_pos = None;
    }

    /// Submit: returns the text, records non-blank entries in history, and clears
    /// the live buffer (history is retained).
    pub fn take(&mut self) -> String {
        let text = self.text();
        if !text.trim().is_empty() {
            self.history.push(text.clone());
        }
        self.lines = vec![Vec::new()];
        self.row = 0;
        self.col = 0;
        self.hist_pos = None;
        text
    }

    fn load(&mut self, s: &str) {
        self.lines = if s.is_empty() {
            vec![Vec::new()]
        } else {
            s.split('\n').map(|l| l.chars().collect()).collect()
        };
        self.row = self.lines.len() - 1;
        self.col = self.lines[self.row].len();
    }

    /// Recall an older history entry (Up arrow at top of buffer).
    pub fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next = match self.hist_pos {
            None => self.history.len() - 1,
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.hist_pos = Some(next);
        let entry = self.history[next].clone();
        self.load(&entry);
    }

    /// Move toward the live buffer (Down arrow); past the newest entry clears.
    pub fn history_next(&mut self) {
        match self.hist_pos {
            None => {}
            Some(i) if i + 1 < self.history.len() => {
                self.hist_pos = Some(i + 1);
                let entry = self.history[i + 1].clone();
                self.load(&entry);
            }
            Some(_) => {
                self.hist_pos = None;
                self.load("");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_empty() {
        let c = Composer::new();
        assert!(c.is_empty());
        assert_eq!(c.text(), "");
        assert_eq!(c.cursor(), (0, 0));
    }

    // User bug: typing past the right edge of the prompt box became invisible
    // — the composer rendered LOGICAL lines unwrapped, sized its height from
    // them, and placed the cursor at the logical column (outside the box).
    // wrap_lines/visual_cursor are the char-exact wrap the renderer uses:
    // each logical line takes len/width + 1 rows, so a fresh input row
    // appears the moment the line fills.
    #[test]
    fn wrap_lines_char_wraps_long_lines() {
        let mut c = Composer::new();
        c.insert_str(&"a".repeat(25));
        assert_eq!(
            c.wrap_lines(10),
            vec!["a".repeat(10), "a".repeat(10), "a".repeat(5)]
        );
        // Empty buffer still renders one row.
        assert_eq!(Composer::new().wrap_lines(10), vec![String::new()]);
    }

    #[test]
    fn wrap_lines_full_line_grows_a_fresh_row() {
        let mut c = Composer::new();
        c.insert_str(&"a".repeat(10));
        // Exactly full → a new empty row appears for continued typing.
        assert_eq!(c.wrap_lines(10), vec!["a".repeat(10), String::new()]);
        assert_eq!(c.visual_cursor(10), (1, 0), "cursor sits on the fresh row");
    }

    #[test]
    fn wrap_lines_covers_multiple_logical_lines() {
        let mut c = Composer::new();
        c.insert_str(&"a".repeat(12));
        c.newline();
        c.insert_str("bb");
        assert_eq!(
            c.wrap_lines(10),
            vec!["a".repeat(10), "aa".to_string(), "bb".to_string()]
        );
        // Cursor at end of "bb": first logical line takes 2 rows, so row 2.
        assert_eq!(c.visual_cursor(10), (2, 2));
    }

    #[test]
    fn visual_cursor_tracks_mid_wrap_position() {
        let mut c = Composer::new();
        c.insert_str(&"a".repeat(25));
        assert_eq!(c.visual_cursor(10), (2, 5));
        // Degenerate width never panics.
        let _ = c.wrap_lines(0);
        let _ = c.visual_cursor(0);
    }

    // Wide chars (CJK, emoji) occupy 2 terminal cells: the wrap and the
    // cursor must count DISPLAY CELLS, not chars — char-based math drifted
    // the cursor and overflowed rows for non-ASCII input.
    #[test]
    fn wide_chars_wrap_by_display_cells() {
        let mut c = Composer::new();
        c.insert_str("日本語のテスト"); // 7 chars × 2 cells = 14 cells
        // width 10 → 5 wide chars (10 cells) per row.
        assert_eq!(
            c.wrap_lines(10),
            vec!["日本語のテ".to_string(), "スト".to_string()]
        );
        // Cursor at end: row 1, after 2 wide chars = cell column 4.
        assert_eq!(c.visual_cursor(10), (1, 4));
    }

    #[test]
    fn wide_char_never_splits_across_a_row_boundary() {
        let mut c = Composer::new();
        c.insert_str("aaa許"); // 3+2 = 5 cells; width 4 leaves 1 cell — 許 must move down
        assert_eq!(c.wrap_lines(4), vec!["aaa".to_string(), "許".to_string()]);
        assert_eq!(c.visual_cursor(4), (1, 2));
    }

    #[test]
    fn exact_full_wide_row_grows_a_fresh_row() {
        let mut c = Composer::new();
        c.insert_str("ab🙂"); // 1+1+2 = 4 cells, exactly full at width 4
        assert_eq!(c.wrap_lines(4), vec!["ab🙂".to_string(), String::new()]);
        assert_eq!(c.visual_cursor(4), (1, 0), "cursor lands on the fresh row");
    }

    #[test]
    fn insert_chars_builds_text() {
        let mut c = Composer::new();
        c.insert_char('h');
        c.insert_char('i');
        assert_eq!(c.text(), "hi");
        assert!(!c.is_empty());
        assert_eq!(c.cursor(), (0, 2));
    }

    #[test]
    fn unicode_insert_is_char_correct() {
        let mut c = Composer::new();
        c.insert_str("héllo");
        c.backspace();
        assert_eq!(c.text(), "héll");
    }

    #[test]
    fn newline_splits_into_two_lines() {
        let mut c = Composer::new();
        c.insert_char('a');
        c.newline();
        c.insert_char('b');
        assert_eq!(c.text(), "a\nb");
        assert_eq!(c.render_lines().len(), 2);
        assert_eq!(c.cursor(), (1, 1));
    }

    #[test]
    fn backspace_within_line() {
        let mut c = Composer::new();
        c.insert_str("ab");
        c.backspace();
        assert_eq!(c.text(), "a");
    }

    #[test]
    fn backspace_joins_lines() {
        let mut c = Composer::new();
        c.insert_char('a');
        c.newline();
        c.backspace();
        assert_eq!(c.text(), "a");
        assert_eq!(c.render_lines().len(), 1);
        assert_eq!(c.cursor(), (0, 1));
    }

    #[test]
    fn insert_str_handles_embedded_newlines() {
        let mut c = Composer::new();
        c.insert_str("line1\nline2");
        assert_eq!(c.text(), "line1\nline2");
        assert_eq!(c.render_lines().len(), 2);
    }

    #[test]
    fn take_returns_clears_and_records_history() {
        let mut c = Composer::new();
        c.insert_str("hello");
        let out = c.take();
        assert_eq!(out, "hello");
        assert!(c.is_empty());
        // history recall brings it back
        c.history_prev();
        assert_eq!(c.text(), "hello");
    }

    #[test]
    fn blank_submissions_are_not_recorded() {
        let mut c = Composer::new();
        c.insert_str("   ");
        let _ = c.take();
        c.history_prev();
        assert_eq!(c.text(), "", "blank entry must not enter history");
    }

    #[test]
    fn mention_prefix_detects_at_token_at_cursor() {
        let mut c = Composer::new();
        c.insert_str("see @src/ma");
        assert_eq!(c.mention_prefix().as_deref(), Some("src/ma"));
        c.insert_char(' ');
        assert_eq!(c.mention_prefix(), None, "a space ends the @token");
    }

    #[test]
    fn complete_mention_replaces_the_token() {
        let mut c = Composer::new();
        c.insert_str("open @ap and go");
        // cursor is at end; move it back to just after "ap"
        for _ in 0.." and go".len() {
            c.move_left();
        }
        c.complete_mention("src/app.rs");
        assert_eq!(c.text(), "open src/app.rs  and go");
    }

    #[test]
    fn set_text_replaces_buffer_cursor_at_end() {
        let mut c = Composer::new();
        c.insert_str("foo");
        c.set_text("/model ");
        assert_eq!(c.text(), "/model ");
        assert_eq!(c.cursor(), (0, 7));
    }

    #[test]
    fn history_prev_and_next_navigate() {
        let mut c = Composer::new();
        c.insert_str("first");
        c.take();
        c.insert_str("second");
        c.take();
        c.history_prev(); // -> second (newest)
        assert_eq!(c.text(), "second");
        c.history_prev(); // -> first
        assert_eq!(c.text(), "first");
        c.history_next(); // -> second
        assert_eq!(c.text(), "second");
        c.history_next(); // -> back to live (empty)
        assert_eq!(c.text(), "");
    }
}
