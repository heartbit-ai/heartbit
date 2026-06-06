//! Startup splash: a beating block-art heart + `heartbit` lettering, rendered
//! by `ui::view` as a full-frame overlay while `App.splash` is `Some(tick)`.
//! Pure functions only — all timing lives in the reducer (`Msg::Tick`).

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Total splash duration in 120ms ticks (~3s — four heartbeats; any key
/// skips, so generous beats are low-cost).
pub const SPLASH_TICKS: u8 = 25;
// ~3s at 120ms/tick — long enough to actually see it (user feedback:
// the original 13 ticks / 1.5s felt too short). Compile-time guard.
const _: () = assert!(SPLASH_TICKS >= 25);

/// Large (systole) heart frame, 7 rows of half-block art.
const HEART_LARGE: [&str; 7] = [
    " ▄▄██▄▄   ▄▄██▄▄ ",
    "█████████████████",
    " ███████████████ ",
    "   ███████████   ",
    "     ███████     ",
    "       ███       ",
    "        ▀        ",
];

/// Small (diastole) frame — 5 rows padded to 7 so the layout never shifts.
const HEART_SMALL: [&str; 7] = [
    "",
    " ▄█▄ ▄█▄ ",
    "█████████",
    " ███████ ",
    "   ███   ",
    "    ▀    ",
    "",
];

/// `heartbit` in half-block letters (H E A R T B I T).
const LETTERING: [&str; 2] = [
    "█ █ █▀▀ ▄▀█ █▀█ ▀█▀ █▄▄ █ ▀█▀",
    "█▀█ ██▄ █▀█ █▀▄  █  █▄█ █  █ ",
];

/// Beat rhythm, period 6: four LARGE (bright) ticks then two SMALL (dim) —
/// cyclic, so the heart beats for the whole splash whatever its duration.
pub(crate) fn is_large(tick: u8) -> bool {
    tick % 6 < 4
}

/// The full splash as centered-ready lines: heart (beat frame by `tick`),
/// lettering, `v{version} · {model}`, and a dim dismissal hint.
pub fn splash_lines(tick: u8, model: &str) -> Vec<Line<'static>> {
    let (heart, heart_style) = if is_large(tick) {
        (
            &HEART_LARGE,
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        (&HEART_SMALL, Style::default().fg(Color::Red))
    };
    let mut lines: Vec<Line<'static>> = heart
        .iter()
        .map(|r| Line::from(Span::styled((*r).to_string(), heart_style)))
        .collect();
    lines.push(Line::raw(""));
    let letter_style = Style::default()
        .fg(Color::Magenta)
        .add_modifier(Modifier::BOLD);
    lines.extend(
        LETTERING
            .iter()
            .map(|r| Line::from(Span::styled((*r).to_string(), letter_style))),
    );
    let dim = Style::default().fg(Color::DarkGray);
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        format!("v{} · {model}", env!("CARGO_PKG_VERSION")),
        dim,
    )));
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "any key",
        dim.add_modifier(Modifier::ITALIC),
    )));
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rhythm_beats_cyclically_for_the_whole_duration() {
        // Period 6: four bright ticks, two dim — the heart keeps beating
        // however long the splash lasts (user feedback: 1.5s was too short).
        for t in [0u8, 3, 6, 9, 12, 18, 24] {
            assert!(is_large(t), "tick {t} must be the LARGE frame");
        }
        for t in [4u8, 5, 10, 11, 16, 22] {
            assert!(!is_large(t), "tick {t} must be the SMALL frame");
        }
    }

    #[test]
    fn frames_never_shift_layout() {
        let large = splash_lines(0, "m/x");
        let small = splash_lines(4, "m/x");
        assert_eq!(large.len(), small.len(), "line count must not change");
        assert!(large.len() >= 12, "heart + lettering + meta lines");
    }

    #[test]
    fn lines_carry_version_model_and_differ_by_frame() {
        let text = |lines: &[ratatui::text::Line]| -> String {
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
        };
        let large = text(&splash_lines(0, "qwen/q3"));
        let small = text(&splash_lines(4, "qwen/q3"));
        assert!(large.contains(env!("CARGO_PKG_VERSION")));
        assert!(large.contains("qwen/q3"));
        assert_ne!(large, small, "beat frames must differ");
    }
}
