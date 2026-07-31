//! Pure parsing/formatting for the `/diff` command: turn the text `main.rs`'s
//! `Effect::GitDiff` handler gathers (`git diff HEAD`'s own output, plus
//! untracked files formatted as synthetic "new file" sections) into the same
//! `DiffLine` shape a tool-call preview uses, so `/diff` renders through the
//! identical `render_diff_lines` path. All I/O (running `git`, listing and
//! reading untracked files) happens in `main.rs`; this module never touches
//! the filesystem or a subprocess, so it's unit-tested without a git repo.

use crate::diff::{DiffLine, parse_unified};

/// Parse the combined working-tree diff text into renderable diff lines.
/// Delegates entirely to [`parse_unified`] — once an untracked file has been
/// formatted by [`format_untracked`], it's indistinguishable from a real
/// unified diff section.
pub fn parse(diff_text: &str) -> Vec<DiffLine> {
    parse_unified(diff_text)
}

/// Format an untracked file's content as a synthetic "new file" unified-diff
/// section (`--- /dev/null` / `+++ b/path` / one hunk of pure additions), so
/// it can be appended to `git diff HEAD`'s output and parsed uniformly by
/// [`parse`]. Leads with a `diff --git` marker — mirroring the section
/// framing real `git diff` emits per file — so concatenating several of
/// these (or appending to a real multi-file diff) still parses each file's
/// own header correctly instead of only recognizing the first.
pub fn format_untracked(path: &str, content: &str) -> String {
    let body_lines: Vec<&str> = if content.is_empty() {
        Vec::new()
    } else {
        content.trim_end_matches('\n').split('\n').collect()
    };
    let mut out = format!(
        "diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n\
         @@ -0,0 +1,{} @@\n",
        body_lines.len()
    );
    for l in &body_lines {
        out.push('+');
        out.push_str(l);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::DiffKind;

    #[test]
    fn parse_delegates_to_parse_unified() {
        let d = parse("@@ -1,1 +1,1 @@\n-a\n+b\n");
        assert_eq!(
            d.iter().map(|l| l.kind).collect::<Vec<_>>(),
            vec![DiffKind::Ctx, DiffKind::Del, DiffKind::Add]
        );
    }

    #[test]
    fn format_untracked_is_an_all_additions_new_file_section() {
        let text = format_untracked("src/new.rs", "fn main() {}\nprintln!(\"hi\");");
        let d = parse(&text);
        // The only non-hunk-header lines are pure additions (no deletions).
        assert!(
            d.iter().all(|l| l.kind != DiffKind::Del),
            "an untracked file has no deletions: {d:?}"
        );
        assert!(
            d.iter()
                .any(|l| l.text == "fn main() {}" && l.kind == DiffKind::Add)
        );
        assert!(
            d.iter()
                .any(|l| l.text == "println!(\"hi\");" && l.kind == DiffKind::Add)
        );
        // The synthetic header itself must not leak into the rendered text.
        assert!(d.iter().all(|l| !l.text.contains("/dev/null")));
    }

    #[test]
    fn format_untracked_empty_file_has_no_lines() {
        let text = format_untracked("empty.txt", "");
        let d = parse(&text);
        assert!(
            d.iter()
                .all(|l| l.kind != DiffKind::Add && l.kind != DiffKind::Del),
            "an empty untracked file adds no content lines: {d:?}"
        );
    }

    #[test]
    fn format_untracked_concatenates_after_a_tracked_hunk_without_leaking_headers() {
        // Simulates the real `/diff` effect handler's output: `git diff
        // HEAD`'s tracked-file text, with an untracked file's synthetic
        // section appended. The tracked section's own `@@` must not stop the
        // untracked section's header from being recognized.
        let tracked = "diff --git a/tracked.rs b/tracked.rs\n--- a/tracked.rs\n\
                        +++ b/tracked.rs\n@@ -1,1 +1,1 @@\n-old\n+new\n";
        let combined = format!("{tracked}{}", format_untracked("brand_new.rs", "hello"));
        let d = parse(&combined);
        assert!(d.iter().any(|l| l.text == "old" && l.kind == DiffKind::Del));
        assert!(d.iter().any(|l| l.text == "new" && l.kind == DiffKind::Add));
        assert!(
            d.iter()
                .any(|l| l.text == "hello" && l.kind == DiffKind::Add)
        );
        assert!(
            d.iter()
                .all(|l| !l.text.contains("brand_new.rs") && !l.text.contains("tracked.rs")),
            "no file header leaked as content: {d:?}"
        );
    }
}
