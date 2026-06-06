//! Derive a compact diff from a file-editing tool's *input* JSON, so the
//! transcript can show "what changed" (the SOTA coding-TUI surface). Pure and
//! table-tested. The data is already in `Cell::Tool.input`:
//!   - `edit`  → `old_string` / `new_string`  (before/after)
//!   - `patch` → `patch_text`                 (already a unified diff)
//!   - `write` → `content`                    (a new file: all additions)
//!
//! Any other tool, or malformed input, yields an empty Vec → the caller falls
//! back to the normal compact tool cell (never panic, never render garbage).

/// One line of a rendered diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffLine {
    pub kind: DiffKind,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
                        });
                    }
                    for l in new.trim_end_matches('\n').split('\n') {
                        lines.push(DiffLine {
                            kind: DiffKind::Add,
                            text: l.to_string(),
                        });
                    }
                    lines
                }
                _ => Vec::new(),
            }
        }
        "patch" => {
            let Some(text) = v.get("patch_text").and_then(|s| s.as_str()) else {
                return Vec::new();
            };
            text.trim_end_matches('\n')
                .split('\n')
                .filter_map(|l| {
                    // Skip file headers (+++/---) and the leading diff/index lines.
                    if l.starts_with("+++")
                        || l.starts_with("---")
                        || l.starts_with("diff ")
                        || l.starts_with("index ")
                    {
                        return None;
                    }
                    let (kind, text) = if let Some(rest) = l.strip_prefix('+') {
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
                        text: text.to_string(),
                    })
                })
                .collect()
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
                })
                .collect()
        }
        _ => Vec::new(),
    }
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
}
