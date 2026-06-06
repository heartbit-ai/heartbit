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
