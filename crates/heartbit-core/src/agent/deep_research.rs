//! The `deep_research` workflow recipe: plan → parallel search/read (each
//! angle agent carries its own websearch/webfetch tools) → cross-verify →
//! synthesize a cited report. Born from a live failure (session 6a245538):
//! asked to "deep research", the agent had no harness to route to, its
//! scraped searches died silently, and it fabricated URLs.

/// Clamp bounds for the `angles` argument.
const MIN_ANGLES: usize = 2;

/// Parse the planning agent's angle list: accepts `1. foo` / `1) foo` /
/// `- foo` / `* foo` lines, trims, drops empties, caps at `max`. Returns the
/// deterministic fallback (the question + a state-of-the-art variant) when
/// fewer than [`MIN_ANGLES`] parse — the plan stage can never fail the run.
fn parse_angles(text: &str, max: usize, question: &str) -> Vec<String> {
    let mut angles: Vec<String> = text
        .lines()
        .map(|l| {
            l.trim()
                .trim_start_matches(|c: char| c.is_ascii_digit())
                .trim_start_matches(['.', ')', '-', '*'])
                .trim()
                .to_string()
        })
        .filter(|l| !l.is_empty())
        .take(max)
        .collect();
    if angles.len() < MIN_ANGLES {
        angles = vec![
            question.to_string(),
            format!("state of the art: {question}"),
        ];
    }
    angles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_angles_accepts_numbered_and_bulleted() {
        let text =
            "1. definition and use cases\n2) core algorithms\n- existing implementations\n* pitfalls\n\n";
        let a = parse_angles(text, 6, "q");
        assert_eq!(
            a,
            vec![
                "definition and use cases",
                "core algorithms",
                "existing implementations",
                "pitfalls"
            ]
        );
    }

    #[test]
    fn parse_angles_caps_at_max() {
        let text = "1. a\n2. b\n3. c\n4. d\n5. e";
        assert_eq!(parse_angles(text, 3, "q").len(), 3);
    }

    #[test]
    fn parse_angles_falls_back_on_garbage() {
        let a = parse_angles("I will now think about this.", 4, "plate solving");
        // One prose line parses as one "angle" — below MIN_ANGLES → fallback.
        assert_eq!(a.len(), 2);
        assert_eq!(a[0], "plate solving");
        assert!(a[1].contains("state of the art"));
        let b = parse_angles("", 4, "q");
        assert_eq!(b.len(), 2);
    }
}
