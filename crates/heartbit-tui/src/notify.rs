//! Desktop notifications via terminal escape sequences.
//!
//! Pure formatting (`sequence`) and sanitizing (`sanitize_field`) plus one thin
//! I/O wrapper (`emit`) called ONLY from the main loop's effect pass — never
//! from the reducer and never from the agent thread.
//!
//! Exactly ONE sequence is sent per terminal: kitty, WezTerm and Ghostty
//! implement both OSC 777 and OSC 9, so sending both notifies twice.

/// Max chars kept per field after sanitizing.
pub(crate) const MAX_FIELD: usize = 120;

/// Which notification sequence this terminal understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Terminal {
    /// `OSC 777 ; notify ; title ; body BEL` — kitty, WezTerm, Ghostty.
    Osc777,
    /// `OSC 9 ; text BEL` — terminals that take 9 but not 777.
    Osc9,
    /// No OSC notification support: fall back to the bell.
    Bell,
}

impl Terminal {
    /// Resolve from `TERM` / `TERM_PROGRAM`.
    pub(crate) fn from_ids(term: Option<&str>, term_program: Option<&str>) -> Self {
        let hay = format!(
            "{} {}",
            term.unwrap_or_default().to_ascii_lowercase(),
            term_program.unwrap_or_default().to_ascii_lowercase()
        );
        if ["kitty", "wezterm", "ghostty", "foot"]
            .iter()
            .any(|t| hay.contains(t))
        {
            Self::Osc777
        } else if ["iterm", "alacritty"].iter().any(|t| hay.contains(t)) {
            Self::Osc9
        } else {
            Self::Bell
        }
    }

    pub(crate) fn from_env() -> Self {
        Self::from_ids(
            std::env::var("TERM").ok().as_deref(),
            std::env::var("TERM_PROGRAM").ok().as_deref(),
        )
    }
}

/// Strip everything that could terminate or re-open an OSC string, plus the `;`
/// field separator, then cap the length. Removes C0 (`< 0x20`), DEL (`0x7f`) and
/// C1 (`U+0080..=U+009F`, which includes ST).
pub(crate) fn sanitize_field(s: &str) -> String {
    s.chars()
        .filter(|c| {
            let n = *c as u32;
            n >= 0x20 && n != 0x7f && !(0x80..=0x9f).contains(&n) && *c != ';'
        })
        .take(MAX_FIELD)
        .collect()
}

/// The exact bytes to write. Fields must already be sanitized.
pub(crate) fn sequence(term: Terminal, title: &str, body: &str) -> String {
    match term {
        Terminal::Osc777 => format!("\x1b]777;notify;{title};{body}\x07"),
        Terminal::Osc9 => format!("\x1b]9;{title}: {body}\x07"),
        Terminal::Bell => "\x07".to_string(),
    }
}

/// Write the notification. Called ONLY from the main loop's effect pass, after
/// `terminal.draw()` returned. Emits OSC + BEL only: nothing here moves the
/// cursor, alters the screen buffer or writes a newline, so the alt-screen frame
/// is byte-identical.
pub(crate) fn emit(title: &str, body: &str) {
    let seq = sequence(
        Terminal::from_env(),
        &sanitize_field(title),
        &sanitize_field(body),
    );
    use std::io::Write;
    let mut out = std::io::stdout();
    let _ = out.write_all(seq.as_bytes());
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_strips_osc_terminators_and_caps_length() {
        // C0, DEL, C1 (incl. U+009C ST) and ';' must not survive: agent-controlled
        // text (tool names, provider errors) reaches the terminal through here.
        // (`\x9c` isn't a legal Rust string escape — string literals only allow
        // `\x00..=\x7f` — so the C1 byte is written as `\u{9c}`, the same
        // codepoint U+009C.)
        assert_eq!(sanitize_field("a\x07b\x1bc\u{9c}d;e\x7f"), "abcde");
        assert_eq!(sanitize_field(&"x".repeat(500)).len(), MAX_FIELD);
    }

    #[test]
    fn sequence_is_exactly_one_per_terminal() {
        assert_eq!(
            sequence(Terminal::Osc777, "T", "B"),
            "\x1b]777;notify;T;B\x07"
        );
        assert_eq!(sequence(Terminal::Osc9, "T", "B"), "\x1b]9;T: B\x07");
        assert_eq!(sequence(Terminal::Bell, "T", "B"), "\x07");
    }

    #[test]
    fn terminal_from_env_maps_known_ids_and_never_double_notifies() {
        // kitty/WezTerm/Ghostty implement BOTH OSC 777 and OSC 9 — pick 777 only.
        assert_eq!(
            Terminal::from_ids(Some("xterm-kitty"), None),
            Terminal::Osc777
        );
        assert_eq!(Terminal::from_ids(None, Some("WezTerm")), Terminal::Osc777);
        assert_eq!(Terminal::from_ids(None, Some("ghostty")), Terminal::Osc777);
        assert_eq!(
            Terminal::from_ids(Some("xterm-256color"), None),
            Terminal::Bell
        );
    }
}
