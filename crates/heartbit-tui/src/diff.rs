//! Derive a compact diff from a file-editing tool's *input* JSON, so the
//! transcript can show "what changed" (the SOTA coding-TUI surface). Pure and
//! table-tested. The data is already in `Cell::Tool.input`:
//!   - `edit`  → `old_string` / `new_string`  (before/after)
//!   - `patch` → `patch_text`                 (already a unified diff)
//!   - `write` → `content`                    (a new file: all additions)
//!
//! Any other tool, or malformed input, yields an empty Vec → the caller falls
//! back to the normal compact tool cell (never panic, never render garbage).
//!
//! Both the `edit` tool's Del/Add lines and `parse_unified` (used by the
//! `patch` tool and `/diff` via `gitdiff::parse`) pair adjacent Del/Add runs
//! of equal length and fill their `emph` ranges via [`word_emphasis`], so a
//! one-word change doesn't read as a whole rewritten line. `write` is pure
//! additions (a new file has no Del run to pair against), so it never gets
//! emphasis — not because it's excluded, but because there's nothing to pair.

use std::ops::Range;

/// One line of a rendered diff.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DiffLine {
    pub kind: DiffKind,
    pub text: String,
    /// Word-level emphasis within `text` (byte ranges): sorted,
    /// non-overlapping, non-empty, and each bound lands on a char boundary.
    /// Empty for context lines and for any Del/Add line that wasn't part of
    /// an equal-length adjacent pairing (mismatched old/new line counts on
    /// an `edit`, `write`'s pure-addition lines which have no Del run to
    /// pair against, or a run past the pairing budget) — such a line renders
    /// to exactly one span, identical to before this field existed.
    #[serde(default)]
    pub emph: Vec<Range<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DiffKind {
    Add,
    Del,
    /// Context / unchanged / hunk header.
    Ctx,
}

/// Produce diff lines for a file-editing tool call, or empty if not applicable.
pub fn diff_lines(tool_name: &str, input_json: &str) -> Vec<DiffLine> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(input_json) else {
        return Vec::new();
    };
    match tool_name {
        "edit" => {
            let old = v.get("old_string").and_then(|s| s.as_str());
            let new = v.get("new_string").and_then(|s| s.as_str());
            match (old, new) {
                (Some(old), Some(new)) => {
                    let mut lines = Vec::new();
                    for l in old.trim_end_matches('\n').split('\n') {
                        lines.push(DiffLine {
                            kind: DiffKind::Del,
                            text: l.to_string(),
                            emph: Vec::new(),
                        });
                    }
                    for l in new.trim_end_matches('\n').split('\n') {
                        lines.push(DiffLine {
                            kind: DiffKind::Add,
                            text: l.to_string(),
                            emph: Vec::new(),
                        });
                    }
                    // The Del run (all of `old`) is immediately followed by
                    // the Add run (all of `new`) by construction above, so
                    // this pairs 1:1 when the line counts match — the most
                    // common diff surface in the transcript gets word
                    // emphasis too, not just `patch`/`/diff`. `write` is
                    // pure Add lines (no Del run ever precedes it), so this
                    // is a no-op there; not called on that arm.
                    pair_and_emphasize(&mut lines);
                    lines
                }
                _ => Vec::new(),
            }
        }
        "patch" => {
            let Some(text) = v.get("patch_text").and_then(|s| s.as_str()) else {
                return Vec::new();
            };
            parse_unified(text)
        }
        "write" => {
            let Some(content) = v.get("content").and_then(|s| s.as_str()) else {
                return Vec::new();
            };
            content
                .trim_end_matches('\n')
                .split('\n')
                .map(|l| DiffLine {
                    kind: DiffKind::Add,
                    text: l.to_string(),
                    emph: Vec::new(),
                })
                .collect()
        }
        _ => Vec::new(),
    }
}

/// Cap on the token-pair DP table for a single line's word diff (64×64):
/// bounds the cost of one [`word_emphasis`] call to a few thousand cell
/// comparisons regardless of how long or dissimilar the two lines are. Past
/// this, the whole trimmed middle is reported as one changed span per side
/// (less precise, but O(1) instead of O(n·m)).
const WORD_DIFF_DP_CAP: usize = 4096;

/// Cap on how many Del/Add lines [`parse_unified`] will attempt to pair for
/// word-level emphasis. The transcript re-renders every cell on every frame
/// (`ui::transcript_lines`), so a pathological multi-thousand-line diff would
/// otherwise re-run one bounded DP per paired line on every redraw. Past this
/// many paired lines, remaining Del/Add runs keep `emph` empty — they still
/// render (as plain colored lines), just without word-level emphasis.
const MAX_PAIRED_LINES: usize = 1000;

/// Parse already-unified-diff text (a `patch` tool's `patch_text`, or the
/// combined text `/diff` builds from `git diff HEAD` plus untracked files —
/// see `gitdiff::parse`) into [`DiffLine`]s, pairing adjacent Del/Add runs of
/// equal length and filling their `emph` via [`word_emphasis`].
///
/// Core's patch parser requires a literal `@@ ` line to recognize unified
/// diff text at all (`patch.rs:425`); text with none takes the legacy
/// (pre-word-emphasis) path unchanged — same line classification, no
/// emphasis — so the `patch` tool's existing behavior cannot regress.
pub fn parse_unified(text: &str) -> Vec<DiffLine> {
    let text = text.trim_end_matches('\n');
    if !text.lines().any(|l| l.starts_with("@@ ")) {
        return legacy_parse(text);
    }
    let raw: Vec<&str> = text.split('\n').collect();
    let mut lines = Vec::new();
    // True at the very start and right after a new file section begins — the
    // window before this section's first hunk, in which EVERY line is
    // metadata (`---`/`+++`/`index`/`new file mode`/`similarity index`/
    // `Binary files … differ`/…), never real hunk content: the unified-diff
    // format guarantees hunk lines only ever appear after an `@@` line, so no
    // per-prefix check is needed here beyond spotting the section boundary —
    // just drop everything until `@@`.
    //
    // A new section is signalled either by a `diff --git` line (git's own
    // `diff` output) or by a `--- `/`+++ ` header PAIR (core's `patch` tool
    // format, which never emits `diff --git` at all — patch.rs:413-421,
    // exercised by its own `patch_multi_file` test). The `+++ ` lookahead on
    // the `--- ` check is required and mirrors core's own rule exactly: a
    // *deleted* line whose content itself starts with `-- ` renders as
    // `--- ...` (leading `-` diff marker + content) and must not be mistaken
    // for a header — only a genuine `--- `/`+++ ` PAIR counts.
    let mut expect_header = true;
    let mut i = 0;
    while i < raw.len() {
        let l = raw[i];
        if l.starts_with("diff ") {
            expect_header = true;
            i += 1;
            continue;
        }
        if l.starts_with("--- ") && raw.get(i + 1).is_some_and(|n| n.starts_with("+++ ")) {
            expect_header = true;
            i += 1;
            continue;
        }
        if l.starts_with("@@") {
            expect_header = false;
            lines.push(DiffLine {
                kind: DiffKind::Ctx,
                text: l.to_string(),
                emph: Vec::new(),
            });
            i += 1;
            continue;
        }
        if expect_header {
            i += 1;
            continue;
        }
        let (kind, txt) = if let Some(rest) = l.strip_prefix('+') {
            (DiffKind::Add, rest)
        } else if let Some(rest) = l.strip_prefix('-') {
            (DiffKind::Del, rest)
        } else {
            (DiffKind::Ctx, l.strip_prefix(' ').unwrap_or(l))
        };
        lines.push(DiffLine {
            kind,
            text: txt.to_string(),
            emph: Vec::new(),
        });
        i += 1;
    }
    pair_and_emphasize(&mut lines);
    lines
}

/// The original (pre-word-emphasis) `patch`-tool parsing: unconditional
/// header stripping (no hunk-position awareness), no emphasis. Used verbatim
/// for hunkless text, matching core's patch parser which requires `@@ ` to
/// recognize a hunk at all — so text that never establishes one takes this
/// exact bit-for-bit fallback rather than the hunk-aware path above.
fn legacy_parse(text: &str) -> Vec<DiffLine> {
    text.split('\n')
        .filter_map(|l| {
            // Skip file headers (+++/---) and the leading diff/index lines.
            if l.starts_with("+++")
                || l.starts_with("---")
                || l.starts_with("diff ")
                || l.starts_with("index ")
            {
                return None;
            }
            let (kind, txt) = if let Some(rest) = l.strip_prefix('+') {
                (DiffKind::Add, rest)
            } else if let Some(rest) = l.strip_prefix('-') {
                (DiffKind::Del, rest)
            } else if l.starts_with("@@") {
                (DiffKind::Ctx, l)
            } else {
                (DiffKind::Ctx, l.strip_prefix(' ').unwrap_or(l))
            };
            Some(DiffLine {
                kind,
                text: txt.to_string(),
                emph: Vec::new(),
            })
        })
        .collect()
}

/// Walk `lines`, pairing each maximal Del run with the Add run immediately
/// following it when they're the same length, and fill both sides' `emph`
/// via [`word_emphasis`]. Unequal-length runs (a real insertion/deletion, not
/// a same-line edit) are left unemphasized — pairing line *N* of a Del run
/// with line *N* of a differently-sized Add run wouldn't be a meaningful
/// correspondence. Bounded by `MAX_PAIRED_LINES` (see its doc comment).
fn pair_and_emphasize(lines: &mut [DiffLine]) {
    let mut i = 0;
    let mut budget = MAX_PAIRED_LINES;
    while i < lines.len() && budget > 0 {
        if lines[i].kind != DiffKind::Del {
            i += 1;
            continue;
        }
        let del_start = i;
        let mut del_end = del_start;
        while del_end < lines.len() && lines[del_end].kind == DiffKind::Del {
            del_end += 1;
        }
        let add_start = del_end;
        let mut add_end = add_start;
        while add_end < lines.len() && lines[add_end].kind == DiffKind::Add {
            add_end += 1;
        }
        let del_len = del_end - del_start;
        let add_len = add_end - add_start;
        if del_len == add_len {
            for k in 0..del_len {
                if budget == 0 {
                    break;
                }
                let (d, a) = word_emphasis(&lines[del_start + k].text, &lines[add_start + k].text);
                lines[del_start + k].emph = d;
                lines[add_start + k].emph = a;
                budget -= 1;
            }
        }
        i = add_end;
    }
}

/// Word-level emphasis between two lines that may differ: tokenize both on
/// word boundaries, trim the common prefix/suffix at token granularity, then
/// diff the remaining middle tokens (bounded DP) so a common token sandwiched
/// between two changes — e.g. `"a b c"` → `"x b y"` — still gets excluded
/// from emphasis. Returns `(empty, empty)` for identical lines.
///
/// Bounded: the middle-token DP is capped at `WORD_DIFF_DP_CAP` cells; past
/// that (a very long, mostly-dissimilar pair of lines), the whole trimmed
/// middle is reported as a single changed span per side instead of doing
/// unbounded work on the UI thread.
pub fn word_emphasis(del: &str, add: &str) -> (Vec<Range<usize>>, Vec<Range<usize>>) {
    if del == add {
        return (Vec::new(), Vec::new());
    }
    let del_tok = tokenize(del);
    let add_tok = tokenize(add);

    let mut prefix = 0;
    while prefix < del_tok.len()
        && prefix < add_tok.len()
        && del[del_tok[prefix].clone()] == add[add_tok[prefix].clone()]
    {
        prefix += 1;
    }
    let del_rest = del_tok.len() - prefix;
    let add_rest = add_tok.len() - prefix;
    let mut suffix = 0;
    while suffix < del_rest
        && suffix < add_rest
        && del[del_tok[del_tok.len() - 1 - suffix].clone()]
            == add[add_tok[add_tok.len() - 1 - suffix].clone()]
    {
        suffix += 1;
    }

    let del_mid = &del_tok[prefix..del_tok.len() - suffix];
    let add_mid = &add_tok[prefix..add_tok.len() - suffix];

    let (del_changed, add_changed) = middle_diff(del, del_mid, add, add_mid);
    (merge_ranges(del_changed), merge_ranges(add_changed))
}

/// Split `s` into maximal runs of "word" (alphanumeric/underscore) or
/// "non-word" characters, so concatenating the token spans reconstructs `s`
/// exactly and every boundary lands on a char boundary (built from
/// `char_indices`, never a raw byte offset).
fn tokenize(s: &str) -> Vec<Range<usize>> {
    let mut tokens = Vec::new();
    let mut start = 0usize;
    let mut cur: Option<bool> = None;
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    for (i, c) in s.char_indices() {
        let w = is_word(c);
        match cur {
            Some(prev) if prev == w => {}
            Some(_) => {
                tokens.push(start..i);
                start = i;
                cur = Some(w);
            }
            None => cur = Some(w),
        }
    }
    if start < s.len() {
        tokens.push(start..s.len());
    }
    tokens
}

/// Which tokens in `del_mid`/`add_mid` are NOT part of a common subsequence
/// (matched by token text), found via a bounded DP (classic LCS, hand-rolled
/// — no `similar` crate, matching this project's other hand-rolled parsers).
/// Falls back to "the whole middle is one changed span per side" once the DP
/// table would exceed `WORD_DIFF_DP_CAP` cells.
// The single-element `vec![start..end]`s below are deliberate: a one-span
// "the whole middle changed" result, not a range being (mis)used to build a
// `Vec` of its elements.
#[allow(clippy::single_range_in_vec_init)]
fn middle_diff(
    del: &str,
    del_mid: &[Range<usize>],
    add: &str,
    add_mid: &[Range<usize>],
) -> (Vec<Range<usize>>, Vec<Range<usize>>) {
    let n = del_mid.len();
    let m = add_mid.len();
    if n == 0 && m == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 0 {
        return (Vec::new(), vec![add_mid[0].start..add_mid[m - 1].end]);
    }
    if m == 0 {
        return (vec![del_mid[0].start..del_mid[n - 1].end], Vec::new());
    }
    if n.saturating_mul(m) > WORD_DIFF_DP_CAP {
        return (
            vec![del_mid[0].start..del_mid[n - 1].end],
            vec![add_mid[0].start..add_mid[m - 1].end],
        );
    }
    // Flat (n+1)×(m+1) DP table, row-major — avoids a `Vec<Vec<_>>` of small
    // allocations for what's already a capped, small table.
    let cols = m + 1;
    let mut dp = vec![0u32; (n + 1) * cols];
    let eq = |i: usize, j: usize| del[del_mid[i].clone()] == add[add_mid[j].clone()];
    for i in 1..=n {
        for j in 1..=m {
            dp[i * cols + j] = if eq(i - 1, j - 1) {
                dp[(i - 1) * cols + (j - 1)] + 1
            } else {
                dp[(i - 1) * cols + j].max(dp[i * cols + (j - 1)])
            };
        }
    }
    let mut del_changed = vec![true; n];
    let mut add_changed = vec![true; m];
    let (mut i, mut j) = (n, m);
    while i > 0 && j > 0 {
        if eq(i - 1, j - 1) {
            del_changed[i - 1] = false;
            add_changed[j - 1] = false;
            i -= 1;
            j -= 1;
        } else if dp[(i - 1) * cols + j] >= dp[i * cols + (j - 1)] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    let del_ranges = del_mid
        .iter()
        .zip(del_changed)
        .filter(|(_, changed)| *changed)
        .map(|(r, _)| r.clone())
        .collect();
    let add_ranges = add_mid
        .iter()
        .zip(add_changed)
        .filter(|(_, changed)| *changed)
        .map(|(r, _)| r.clone())
        .collect();
    (del_ranges, add_ranges)
}

/// Merge adjacent (touching) token ranges into contiguous spans, so a
/// multi-token change highlights as one span rather than several
/// back-to-back ones. Output is sorted, non-overlapping and non-empty.
fn merge_ranges(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    ranges.sort_by_key(|r| r.start);
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for r in ranges {
        if r.is_empty() {
            continue;
        }
        if let Some(last) = merged.last_mut()
            && last.end == r.start
        {
            last.end = r.end;
            continue;
        }
        merged.push(r);
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(d: &[DiffLine]) -> Vec<DiffKind> {
        d.iter().map(|l| l.kind).collect()
    }

    #[test]
    fn edit_yields_del_then_add_blocks() {
        let d = diff_lines(
            "edit",
            r#"{"file_path":"f","old_string":"a\nb","new_string":"a\nc"}"#,
        );
        assert_eq!(
            kinds(&d),
            vec![DiffKind::Del, DiffKind::Del, DiffKind::Add, DiffKind::Add]
        );
        assert_eq!(d[0].text, "a");
        assert_eq!(d[1].text, "b");
        assert_eq!(d[3].text, "c");
    }

    #[test]
    fn patch_parses_unified_diff_by_leading_char() {
        let patch = "--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n ctx\n-old line\n+new line\n";
        let d = diff_lines(
            "patch",
            &serde_json::json!({ "patch_text": patch }).to_string(),
        );
        // file headers dropped; @@ kept as ctx; ctx/-/+ classified
        assert_eq!(
            kinds(&d),
            vec![DiffKind::Ctx, DiffKind::Ctx, DiffKind::Del, DiffKind::Add]
        );
        assert_eq!(d[2].text, "old line");
        assert_eq!(d[3].text, "new line");
        assert!(d.iter().all(|l| !l.text.starts_with("+++")));
    }

    #[test]
    fn write_is_all_additions() {
        let d = diff_lines("write", r#"{"file_path":"f","content":"line1\nline2"}"#);
        assert_eq!(kinds(&d), vec![DiffKind::Add, DiffKind::Add]);
        assert_eq!(d[0].text, "line1");
        // Pure additions never form a Del run, so `pair_and_emphasize` (which
        // `edit` now also runs) is a no-op here — not an exclusion, just
        // nothing to pair against.
        assert!(d.iter().all(|l| l.emph.is_empty()));
    }

    #[test]
    fn non_editing_tool_yields_empty() {
        assert!(diff_lines("bash", r#"{"command":"ls"}"#).is_empty());
        assert!(diff_lines("read", r#"{"file_path":"f"}"#).is_empty());
    }

    #[test]
    fn malformed_or_missing_fields_yield_empty_not_panic() {
        assert!(diff_lines("edit", "not json").is_empty());
        assert!(diff_lines("edit", r#"{"file_path":"f"}"#).is_empty()); // no old/new
        assert!(diff_lines("patch", r#"{"x":1}"#).is_empty()); // no patch_text
        assert!(diff_lines("write", "{}").is_empty());
    }

    #[test]
    fn single_token_change_emphasises_only_that_token() {
        let (del, add) = word_emphasis("let x = 1;", "let x = 2;");
        assert_eq!(del.len(), 1);
        assert_eq!(add.len(), 1);
        assert_eq!(&"let x = 1;"[del[0].clone()], "1");
        assert_eq!(&"let x = 2;"[add[0].clone()], "2");
    }

    #[test]
    fn identical_lines_yield_no_emphasis() {
        let (del, add) = word_emphasis("same", "same");
        assert!(del.is_empty() && add.is_empty());
    }

    #[test]
    fn two_disjoint_changes_are_both_emphasised() {
        let (del, _) = word_emphasis("a b c", "x b y");
        assert_eq!(del.len(), 2, "both ends changed: {del:?}");
        // Ranges are sorted, non-overlapping and on char boundaries.
        assert!(del.windows(2).all(|w| w[0].end <= w[1].start));
    }

    #[test]
    fn emphasis_ranges_are_char_boundaries_for_multibyte_text() {
        let (del, add) = word_emphasis("héllo wörld", "héllo tërre");
        for (s, rs) in [("héllo wörld", &del), ("héllo tërre", &add)] {
            for r in rs.iter() {
                assert!(s.is_char_boundary(r.start) && s.is_char_boundary(r.end));
            }
        }
    }

    #[test]
    fn del_line_starting_with_a_comment_dash_is_not_dropped_as_a_file_header() {
        // "--- a/x" is a header, but a deleted line whose content starts with
        // "--" is real content and must survive.
        let d = parse_unified("@@ -1,1 +1,1 @@\n---- a comment\n+ok\n");
        assert!(d.iter().any(|l| l.text.contains("a comment")));
    }

    #[test]
    fn hunkless_patch_text_takes_the_legacy_path_unchanged() {
        // Core's patch parser requires "@@ " (patch.rs:425), so hunkless text
        // must behave exactly as before this task.
        let d = parse_unified("+added\n-removed\n");
        assert_eq!(d.len(), 2);
        assert!(d.iter().all(|l| l.emph.is_empty()));
    }

    #[test]
    fn multi_file_diff_headers_are_dropped_per_file_section() {
        // A realistic multi-file `git diff` concatenates several
        // `diff --git`/`---`/`+++`/`@@` groups. Each file's own header must
        // be recognized, not just the first (the naive "seen any @@ yet"
        // rule would leak the second file's `--- a/f2`/`+++ b/f2` as content).
        let text = "diff --git a/f1 b/f1\nindex 111..222 100644\n--- a/f1\n+++ b/f1\n\
                     @@ -1,1 +1,1 @@\n-one\n+1\n\
                     diff --git a/f2 b/f2\nindex 333..444 100644\n--- a/f2\n+++ b/f2\n\
                     @@ -1,1 +1,1 @@\n-two\n+2\n";
        let d = parse_unified(text);
        assert!(
            d.iter()
                .all(|l| !l.text.contains("a/f1") && !l.text.contains("a/f2")),
            "file headers of BOTH sections must be dropped: {d:?}"
        );
        assert!(d.iter().any(|l| l.text == "one"));
        assert!(d.iter().any(|l| l.text == "two"));
    }

    #[test]
    fn multi_file_patch_text_without_diff_git_lines_drops_both_headers() {
        // Core's OWN `patch` tool takes only `patch_text` and never emits a
        // `diff --git` line — multi-file patches are just `--- `/`+++ ` PAIRS
        // back to back (patch.rs:413-421, exercised by core's own
        // `patch_multi_file` test at patch.rs:1058). Mirrors that fixture
        // exactly (two files, one hunk each, no `diff --git` anywhere).
        let text = "--- a/file1.txt\n+++ b/file1.txt\n@@ -1 +1 @@\n-hello\n+HELLO\n\
                     --- a/file2.txt\n+++ b/file2.txt\n@@ -1 +1 @@\n-world\n+WORLD\n";
        let d = parse_unified(text);
        assert!(
            d.iter()
                .all(|l| !l.text.contains("file1.txt") && !l.text.contains("file2.txt")),
            "the second file's `--- `/`+++ ` header must not leak as Del/Add \
             content lines: {d:?}"
        );
        assert_eq!(
            kinds(&d),
            vec![
                DiffKind::Ctx,
                DiffKind::Del,
                DiffKind::Add,
                DiffKind::Ctx,
                DiffKind::Del,
                DiffKind::Add,
            ],
            "{d:?}"
        );
        assert!(
            d.iter()
                .any(|l| l.text == "hello" && l.kind == DiffKind::Del)
        );
        assert!(
            d.iter()
                .any(|l| l.text == "HELLO" && l.kind == DiffKind::Add)
        );
        assert!(
            d.iter()
                .any(|l| l.text == "world" && l.kind == DiffKind::Del)
        );
        assert!(
            d.iter()
                .any(|l| l.text == "WORLD" && l.kind == DiffKind::Add)
        );
    }

    #[test]
    fn word_emphasis_bounds_a_pathological_long_dissimilar_line() {
        // No common tokens at all → prefix/suffix trim finds nothing, the
        // 64×64 DP cap is exceeded (199×199 tokens), so the bounded fallback
        // reports the whole line as one changed span per side — still O(1),
        // never panics or hangs the render.
        let del: String = (0..100)
            .map(|i| format!("d{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let add: String = (0..100)
            .map(|i| format!("a{i}"))
            .collect::<Vec<_>>()
            .join(";");
        let (dr, ar) = word_emphasis(&del, &add);
        assert_eq!(dr, vec![0..del.len()]);
        assert_eq!(ar, vec![0..add.len()]);
    }

    #[test]
    fn pairing_is_bounded_so_a_huge_diff_cannot_block_the_ui_thread() {
        let mut text = String::from("@@ -1,1 +1,1 @@\n");
        for i in 0..(MAX_PAIRED_LINES + 200) {
            text.push_str(&format!("-old{i}\n+new{i}\n"));
        }
        let d = parse_unified(&text);
        // First pair is well within the pairing budget → emphasized.
        assert!(
            !d[1].emph.is_empty(),
            "first del line should be emphasized: {:?}",
            d[1]
        );
        // Far beyond the pairing budget → the cap must have kicked in.
        assert!(
            d.iter()
                .any(|l| l.kind != DiffKind::Ctx && l.emph.is_empty()),
            "pairing must stop somewhere — a huge diff can't run unbounded DP on every redraw"
        );
    }
}
