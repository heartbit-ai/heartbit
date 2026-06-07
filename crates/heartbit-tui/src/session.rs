//! Session persistence: save/restore the transcript (so a long session can be
//! revisited) and export it to Markdown (so it can be shared). Files live under
//! `<config-dir>/sessions/<id>.json`. The Markdown rendering is pure & tested.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::cells::{Cell, ToolStatus};

/// A persisted session: its transcript plus a creation timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub created: String,
    pub history: Vec<Cell>,
}

/// Lightweight metadata for the `/resume` picker (no full history loaded).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionMeta {
    pub id: String,
    pub preview: String,
    pub turns: usize,
}

/// The sessions directory (`<config-dir>/sessions`), created on demand.
pub fn sessions_dir() -> PathBuf {
    crate::config::config_path()
        .parent()
        .map(|p| p.join("sessions"))
        .unwrap_or_else(|| PathBuf::from("sessions"))
}

/// Persist a session's transcript as JSON (best-effort; skips an empty history).
pub fn save(dir: &Path, session: &Session) -> std::io::Result<()> {
    if session.history.is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(dir)?;
    let body = serde_json::to_string(session).map_err(std::io::Error::other)?;
    std::fs::write(dir.join(format!("{}.json", session.id)), body)
}

/// Load a session by id.
pub fn load(dir: &Path, id: &str) -> std::io::Result<Session> {
    let body = std::fs::read_to_string(dir.join(format!("{id}.json")))?;
    serde_json::from_str(&body).map_err(std::io::Error::other)
}

/// List saved sessions (most recent first), with a one-line preview.
pub fn list(dir: &Path) -> Vec<SessionMeta> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut metas: Vec<(std::time::SystemTime, SessionMeta)> = Vec::new();
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("json") {
            continue;
        }
        let mtime = e
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        if let Ok(s) = std::fs::read_to_string(&path)
            && let Ok(session) = serde_json::from_str::<Session>(&s)
        {
            let preview = session
                .history
                .iter()
                .find_map(|c| match c {
                    Cell::User(t) => Some(t.lines().next().unwrap_or("").to_string()),
                    _ => None,
                })
                .unwrap_or_else(|| "(empty)".into());
            let turns = session
                .history
                .iter()
                .filter(|c| matches!(c, Cell::User(_)))
                .count();
            metas.push((
                mtime,
                SessionMeta {
                    id: session.id,
                    preview,
                    turns,
                },
            ));
        }
    }
    metas.sort_by_key(|(t, _)| std::cmp::Reverse(*t)); // most-recent first
    metas.into_iter().map(|(_, m)| m).collect()
}

/// One saved handoff brief (purpose-tailored session bridge).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandoffMeta {
    /// File name (carries the date + purpose slug).
    pub file_name: String,
    /// Absolute path of the brief.
    pub path: PathBuf,
    /// First non-empty line — shown in the picker.
    pub preview: String,
}

/// The handoff-briefs directory (`<config-dir>/handoffs`) — disposable briefs,
/// deliberately OUTSIDE any workspace (no doc rot in repos).
pub fn handoffs_dir() -> PathBuf {
    crate::config::config_path()
        .parent()
        .map(|p| p.join("handoffs"))
        .unwrap_or_else(|| PathBuf::from("handoffs"))
}

/// List handoff briefs, most recent first, with a one-line preview.
pub fn list_handoffs(dir: &Path) -> Vec<HandoffMeta> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut metas: Vec<(std::time::SystemTime, HandoffMeta)> = Vec::new();
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("md") {
            continue;
        }
        let mtime = e
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        let preview = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| !l.trim().is_empty())
                    .map(|l| l.trim_start_matches('#').trim().to_string())
            })
            .unwrap_or_else(|| "(empty)".into());
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        metas.push((
            mtime,
            HandoffMeta {
                file_name,
                path,
                preview,
            },
        ));
    }
    metas.sort_by_key(|(t, _)| std::cmp::Reverse(*t));
    metas.into_iter().map(|(_, m)| m).collect()
}

/// Write a DETERMINISTIC emergency handoff brief on a terminal run failure —
/// no LLM call (the provider may be the very thing that failed, e.g. credits
/// exhausted). Captures the error, the user's requests, and a pointer to the
/// saved session so a fresh session can continue deliberately.
pub fn write_emergency_brief(
    dir: &Path,
    session_id: &str,
    error: &str,
    history: &[Cell],
) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let mut body = String::from("# Emergency handoff (run failed)\n\n");
    body.push_str(&format!("## Why\nThe run died mid-work: {error}\n\n"));
    body.push_str("## User requests this session\n");
    let mut any = false;
    for c in history {
        if let Cell::User(t) = c {
            any = true;
            body.push_str(&format!("- {}\n", t.lines().next().unwrap_or("")));
        }
    }
    if !any {
        body.push_str("- (none recorded)\n");
    }
    body.push_str(&format!(
        "\n## Pointers\n- Full session transcript: `{}/{session_id}.json` (use /resume)\n- \
         Trace: `{}/{session_id}.trace.jsonl`\n\n## Next steps\nResume in a fresh session: \
         review the last assistant/tool activity in the transcript, verify the working tree \
         state (`git status`), and continue the most recent user request.\n",
        sessions_dir().display(),
        sessions_dir().display(),
    ));
    let path = dir.join(format!("emergency-{session_id}.md"));
    std::fs::write(&path, body)?;
    Ok(path)
}

/// Cap on persisted prompt-history entries per directory (last N kept).
const PROMPT_HISTORY_MAX: usize = 500;

/// Stable FNV-1a 64 hash — keys the per-directory prompt-history file by the
/// working directory path (stable across runs and toolchains, unlike
/// `DefaultHasher`).
fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The persistent prompt-history file for `cwd`
/// (`<config-dir>/prompt_history/<fnv-of-path>.jsonl`, one JSON string per line
/// — robust to multi-line prompts). Per-directory by design: ↑ in a project
/// recalls THAT project's prompts.
pub fn prompt_history_file(cwd: &Path) -> PathBuf {
    let key = format!("{:016x}", fnv1a64(&cwd.to_string_lossy()));
    crate::config::config_path()
        .parent()
        .map(|p| p.join("prompt_history").join(format!("{key}.jsonl")))
        .unwrap_or_else(|| PathBuf::from(format!("prompt_history-{key}.jsonl")))
}

/// Load the persisted prompts (older first, last [`PROMPT_HISTORY_MAX`] kept).
pub fn load_prompt_history(file: &Path) -> Vec<String> {
    let Ok(body) = std::fs::read_to_string(file) else {
        return Vec::new();
    };
    let mut entries: Vec<String> = body
        .lines()
        .filter_map(|l| serde_json::from_str::<String>(l).ok())
        .collect();
    if entries.len() > PROMPT_HISTORY_MAX {
        entries.drain(..entries.len() - PROMPT_HISTORY_MAX);
    }
    entries
}

/// Append one prompt (best-effort; creates the directory on first use).
pub fn append_prompt_history(file: &Path, prompt: &str) -> std::io::Result<()> {
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir)?;
    }
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(file)?;
    writeln!(
        f,
        "{}",
        serde_json::to_string(prompt).map_err(std::io::Error::other)?
    )
}

/// Render a transcript as Markdown (pure) for `/export`.
pub fn to_markdown(history: &[Cell]) -> String {
    let mut out = String::from("# Heartbit session\n\n");
    for cell in history {
        match cell {
            Cell::User(t) => {
                out.push_str("## 🧑 You\n\n");
                out.push_str(t);
                out.push_str("\n\n");
            }
            Cell::Agent(t) => {
                out.push_str(t);
                out.push_str("\n\n");
            }
            Cell::Tool {
                name,
                status,
                duration_ms,
                ..
            } => {
                let mark = match status {
                    ToolStatus::Ok => "✓",
                    ToolStatus::Failed => "✗",
                    ToolStatus::Running => "⏳",
                };
                let ms = duration_ms.map(|m| format!(" ({m}ms)")).unwrap_or_default();
                out.push_str(&format!("- `{mark} {name}{ms}`\n"));
            }
            Cell::Notice(t) => out.push_str(&format!("> _{t}_\n\n")),
            Cell::Reasoning(t) => {
                out.push_str("<details><summary>💭 reasoning</summary>\n\n");
                out.push_str(t);
                out.push_str("\n\n</details>\n\n");
            }
            // Markdown can't carry the card's colors — export the plain table.
            Cell::Stats { label, stats } => {
                out.push_str(&format!(
                    "**stats — {label}**\n\n```\n{}```\n\n",
                    stats.render()
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Cell> {
        vec![
            Cell::User("fix the bug".into()),
            Cell::Tool {
                name: "edit".into(),
                input: "{}".into(),
                status: ToolStatus::Ok,
                output: Some("ok".into()),
                duration_ms: Some(5),
                agent: None,
            },
            Cell::Agent("Done — fixed it.".into()),
            Cell::Notice("interrupted".into()),
            Cell::Stats {
                label: "t-9".into(),
                stats: Box::new(crate::trace_stats::TraceStats::default()),
            },
        ]
    }

    #[test]
    fn prompt_history_roundtrips_multiline() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("h.jsonl");
        append_prompt_history(&f, "ligne un\nligne deux").unwrap();
        append_prompt_history(&f, "second prompt").unwrap();
        let loaded = load_prompt_history(&f);
        assert_eq!(
            loaded,
            vec![
                "ligne un\nligne deux".to_string(),
                "second prompt".to_string()
            ]
        );
    }

    #[test]
    fn prompt_history_caps_at_last_max() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("h.jsonl");
        for i in 0..(PROMPT_HISTORY_MAX + 10) {
            append_prompt_history(&f, &format!("p{i}")).unwrap();
        }
        let loaded = load_prompt_history(&f);
        assert_eq!(loaded.len(), PROMPT_HISTORY_MAX);
        assert_eq!(
            loaded.last().unwrap(),
            &format!("p{}", PROMPT_HISTORY_MAX + 9)
        );
        assert_eq!(
            loaded.first().unwrap(),
            "p10",
            "oldest beyond the cap dropped"
        );
    }

    #[test]
    fn prompt_history_missing_file_is_empty() {
        assert!(load_prompt_history(Path::new("/nonexistent/h.jsonl")).is_empty());
    }

    #[test]
    fn prompt_history_is_per_directory() {
        let a = prompt_history_file(Path::new("/tmp/project-a"));
        let b = prompt_history_file(Path::new("/tmp/project-b"));
        assert_ne!(a, b, "each directory gets its own history file");
    }

    #[test]
    fn list_handoffs_newest_first_with_preview() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("old.md"), "# Purpose: refactor\nbody").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(dir.path().join("new.md"), "\n# Purpose: prototype\nbody").unwrap();
        std::fs::write(dir.path().join("ignored.txt"), "not a brief").unwrap();
        let briefs = list_handoffs(dir.path());
        assert_eq!(briefs.len(), 2, "only .md files list");
        assert_eq!(briefs[0].file_name, "new.md", "newest first");
        assert_eq!(briefs[0].preview, "Purpose: prototype");
    }

    #[test]
    fn list_handoffs_missing_dir_is_empty() {
        assert!(list_handoffs(std::path::Path::new("/nonexistent/x")).is_empty());
    }

    #[test]
    fn emergency_brief_is_deterministic_and_pointed() {
        let dir = tempfile::tempdir().unwrap();
        let history = vec![
            Cell::User("add the /health endpoint".into()),
            Cell::Agent("working on it".into()),
        ];
        let path = write_emergency_brief(
            dir.path(),
            "sess-42",
            "API error (402): out of credits",
            &history,
        )
        .unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("402"), "carries the error");
        assert!(
            body.contains("add the /health endpoint"),
            "carries the user requests"
        );
        assert!(body.contains("sess-42"), "points at the session artifacts");
        assert!(
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .contains("sess-42"),
            "file named by session"
        );
    }

    #[test]
    fn markdown_export_covers_all_cell_kinds() {
        let md = to_markdown(&sample());
        assert!(md.contains("## 🧑 You"));
        assert!(md.contains("fix the bug"));
        assert!(md.contains("`✓ edit (5ms)`"));
        assert!(md.contains("Done — fixed it."));
        assert!(md.contains("> _interrupted_"));
        // The stats card exports as the plain fenced table.
        assert!(md.contains("**stats — t-9**"), "{md}");
        assert!(md.contains("tools") || md.contains("records"), "{md}");
    }

    #[test]
    fn save_load_roundtrips_and_lists_with_preview() {
        let dir = tempfile::tempdir().unwrap();
        let s = Session {
            id: "abc123".into(),
            created: "now".into(),
            history: sample(),
        };
        save(dir.path(), &s).unwrap();
        let loaded = load(dir.path(), "abc123").unwrap();
        assert_eq!(loaded.history.len(), 5);
        let metas = list(dir.path());
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].id, "abc123");
        assert_eq!(metas[0].preview, "fix the bug");
        assert_eq!(metas[0].turns, 1);
    }

    #[test]
    fn save_skips_empty_history() {
        let dir = tempfile::tempdir().unwrap();
        save(
            dir.path(),
            &Session {
                id: "empty".into(),
                created: "now".into(),
                history: vec![],
            },
        )
        .unwrap();
        assert!(list(dir.path()).is_empty());
    }
}
