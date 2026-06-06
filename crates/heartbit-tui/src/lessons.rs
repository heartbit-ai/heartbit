#![allow(dead_code)] // TODO(lessons): remove once Task 3 wires /learn

//! Learned lessons (self-improvement rung 2): `/learn` distills `/analyze`
//! diagnoses into `<config-dir>/lessons.md`, which is injected into the
//! agent's system prompt at startup. The agent can only touch the WORKSPACE
//! (builtins reject absolute paths), so the edge stages the file into cwd
//! and commits it back after a digest-guarded validation.
//! Spec: docs/superpowers/specs/2026-06-06-tui-learned-rules-design.md.

use std::path::{Path, PathBuf};

/// Hard cap on the lessons file — the whole point is a SMALL standing prompt.
pub const LESSONS_MAX_BYTES: usize = 16 * 1024;
/// Required first line of the lessons file.
pub const LESSONS_HEADING: &str = "# heartbit lessons";
/// The cwd staging name `/learn` routes through (gitignored).
pub const STAGED_LESSONS: &str = "heartbit-lessons.md";
/// Initial content when no global lessons exist yet.
pub const LESSONS_TEMPLATE: &str = "# heartbit lessons\n\
<!-- distilled by /learn from /analyze diagnoses; one line per lesson; edit freely -->\n";

/// The global lessons file: `<config-dir>/lessons.md`, next to `tui.toml`.
pub fn lessons_path() -> PathBuf {
    crate::config::config_path()
        .parent()
        .map(|p| p.join("lessons.md"))
        .unwrap_or_else(|| PathBuf::from("lessons.md"))
}

/// Count list-item lessons (lines starting with `- `).
pub fn lesson_count(content: &str) -> usize {
    content
        .lines()
        .filter(|l| l.trim_start().starts_with("- "))
        .count()
}

/// Load the global lessons for prompt injection. `None` when absent, empty,
/// or over the cap (a standing prompt must stay small).
pub fn load_lessons() -> Option<String> {
    load_lessons_from(&lessons_path())
}

pub(crate) fn load_lessons_from(path: &Path) -> Option<String> {
    let body = std::fs::read_to_string(path).ok()?;
    if body.trim().is_empty() || body.len() > LESSONS_MAX_BYTES {
        return None;
    }
    Some(body)
}

/// Validate a staged lessons file before committing: exists, non-empty,
/// under the cap, starts with the heading. Returns the lesson count.
pub fn validate_staged(path: &Path) -> Result<usize, String> {
    let body =
        std::fs::read_to_string(path).map_err(|e| format!("staged lessons unreadable: {e}"))?;
    if body.trim().is_empty() {
        return Err("staged lessons are empty".into());
    }
    if body.len() > LESSONS_MAX_BYTES {
        return Err(format!(
            "staged lessons too large ({} bytes > {LESSONS_MAX_BYTES})",
            body.len()
        ));
    }
    if !body.starts_with(LESSONS_HEADING) {
        return Err(format!(
            "staged lessons must start with the `{LESSONS_HEADING}` heading"
        ));
    }
    Ok(lesson_count(&body))
}

/// Commit the staged file to the global lessons path (atomic, 0600 —
/// the config.rs temp+rename pattern).
pub fn commit_lessons(staged: &Path) -> std::io::Result<()> {
    commit_lessons_to(staged, &lessons_path())
}

pub(crate) fn commit_lessons_to(staged: &Path, global: &Path) -> std::io::Result<()> {
    use std::io::Write;
    let body = std::fs::read(staged)?;
    if let Some(parent) = global.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = global.with_extension("tmp");
    {
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&tmp)?;
        f.write_all(&body)?;
    }
    // rename preserves the temp file's 0600 mode → final file is 0600.
    std::fs::rename(&tmp, global)
}

/// Content digest for the changed-since-staging guard (std DefaultHasher —
/// not cryptographic, just "did the agent rewrite the file").
pub fn file_digest(path: &Path) -> Option<u64> {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let body = std::fs::read(path).ok()?;
    let mut h = DefaultHasher::new();
    body.hash(&mut h);
    Some(h.finish())
}

/// The `/learn` task template (a prompt const, like `/analyze`'s — the
/// command knows the guidance is needed every time).
pub fn build_learn_prompt(staged: &str, diagnoses: &[String]) -> String {
    let diags = diagnoses
        .iter()
        .map(|d| format!("- {d}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"Distill the diagnosis report(s) below into the persistent lessons file.

## Inputs (current directory)
- Lessons file (REWRITE this one): {staged}
- Diagnosis report(s) from /analyze (read them all):
{diags}

## Your job
1. Read {staged} (the tool's current learned lessons) and every diagnosis report.
2. Distill the reports' Recommendations into durable, GENERAL lessons about how
   this tool should operate — not session trivia, not one-off facts.
3. Merge with the existing lessons: dedupe, drop anything stale or contradicted,
   keep the strongest formulation of each idea.
4. REWRITE {staged} with the `write` tool. Requirements:
   - first line stays exactly `{heading}`
   - at most 25 lessons, ONE line each, as `- ` list items, ranked by impact
   - imperative voice ("Prefer X over Y when Z"), concrete enough to act on
5. Answer with a 2-3 line summary of what changed (added/merged/dropped).

Use only workspace-relative paths. Do not touch any other file.
"#,
        heading = LESSONS_HEADING,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lessons_path_is_next_to_the_config() {
        let p = lessons_path();
        assert!(p.ends_with("lessons.md"), "got {p:?}");
        assert_eq!(
            p.parent(),
            crate::config::config_path().parent(),
            "lessons live next to tui.toml"
        );
    }

    #[test]
    fn template_has_heading_and_zero_lessons() {
        assert!(LESSONS_TEMPLATE.starts_with(LESSONS_HEADING));
        assert_eq!(lesson_count(LESSONS_TEMPLATE), 0);
    }

    #[test]
    fn lesson_count_counts_list_items_only() {
        let s = "# heartbit lessons\n<!-- meta -->\n- one\n- two\nprose\n- three\n";
        assert_eq!(lesson_count(s), 3);
    }

    #[test]
    fn validate_staged_rejects_missing_empty_overcap_and_wrong_heading() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("heartbit-lessons.md");
        assert!(validate_staged(&p).is_err(), "missing");
        std::fs::write(&p, "").unwrap();
        assert!(validate_staged(&p).is_err(), "empty");
        std::fs::write(&p, "no heading\n- x\n").unwrap();
        assert!(validate_staged(&p).unwrap_err().contains("heading"));
        let big = format!("{LESSONS_HEADING}\n{}", "x".repeat(LESSONS_MAX_BYTES));
        std::fs::write(&p, big).unwrap();
        assert!(validate_staged(&p).unwrap_err().contains("large"));
        std::fs::write(&p, format!("{LESSONS_HEADING}\n- a\n- b\n")).unwrap();
        assert_eq!(validate_staged(&p).unwrap(), 2);
    }

    #[test]
    fn commit_writes_atomically_with_0600() {
        let dir = tempfile::tempdir().unwrap();
        let staged = dir.path().join("heartbit-lessons.md");
        let global = dir.path().join("lessons.md");
        std::fs::write(&staged, format!("{LESSONS_HEADING}\n- a\n")).unwrap();
        commit_lessons_to(&staged, &global).unwrap();
        let body = std::fs::read_to_string(&global).unwrap();
        assert!(body.contains("- a"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&global).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn digest_changes_when_content_changes() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.md");
        std::fs::write(&p, "one").unwrap();
        let d1 = file_digest(&p);
        std::fs::write(&p, "two").unwrap();
        let d2 = file_digest(&p);
        assert!(d1.is_some() && d2.is_some() && d1 != d2);
        assert_eq!(file_digest(&dir.path().join("absent")), None);
    }

    #[test]
    fn load_lessons_respects_cap_and_absence() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("lessons.md");
        assert!(load_lessons_from(&p).is_none(), "absent");
        std::fs::write(&p, "").unwrap();
        assert!(load_lessons_from(&p).is_none(), "empty");
        std::fs::write(&p, format!("{LESSONS_HEADING}\n- a\n")).unwrap();
        assert!(load_lessons_from(&p).is_some());
        std::fs::write(&p, "x".repeat(LESSONS_MAX_BYTES + 1)).unwrap();
        assert!(load_lessons_from(&p).is_none(), "over cap");
    }

    #[test]
    fn learn_prompt_embeds_staged_diagnoses_cap_and_rewrite() {
        let p = build_learn_prompt(
            "heartbit-lessons.md",
            &[
                "heartbit-diagnosis-s1.md".into(),
                "heartbit-diagnosis-s2.md".into(),
            ],
        );
        assert!(p.contains("heartbit-lessons.md"));
        assert!(p.contains("heartbit-diagnosis-s1.md"));
        assert!(p.contains("heartbit-diagnosis-s2.md"));
        assert!(p.to_lowercase().contains("rewrite"));
        assert!(p.contains("25"), "lesson cap stated");
        assert!(p.contains(LESSONS_HEADING));
        assert!(!p.contains("/.config/"), "workspace-relative paths only");
    }
}
