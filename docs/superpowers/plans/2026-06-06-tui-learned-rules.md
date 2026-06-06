# TUI Learned Lessons (Rung 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `/learn` distills `/analyze` diagnoses into `~/.config/heartbit/lessons.md`, committed via a digest-guarded turn-idle state machine, and injected into the agent's system prompt at startup.

**Architecture:** New `lessons.rs` owns the file lifecycle (load/validate/commit/digest/prompt). `/learn` mirrors `/analyze`'s stage-through-cwd shape (builtins reject absolute paths); the reducer tracks `learning: Option<u64>` and commits on the turn-idle `LlmDone{had_tool_calls:false}` (NOT `RunCompleted` — that only fires at session end); Esc clears the flag before the synthetic LlmDone can land. Injection rides the existing AGENTS.md `instructions` composition in `build_engine`.

**Tech Stack:** Rust edition 2024; std only (DefaultHasher for digests); existing TUI patterns. Zero heartbit-core changes.

**Spec:** `docs/superpowers/specs/2026-06-06-tui-learned-rules-design.md`

---

## Verified ground truth

- Atomic 0600 write pattern: `config.rs::write_secret` (OpenOptionsExt mode 0o600, truncate) + `rename` preserves mode (`config.rs:132-148,182-192`). `config_path()` resolves the config dir.
- Turn-idle: `app.rs:580-613` `Msg::LlmDone` arm — `if !had_tool_calls { self.running = false; self.agents_settle(); }`. The commit hook goes there.
- Esc: `app.rs:1277 fn interrupt()` pushes `Effect::Interrupt`, finalizes, sets `running=false`. Clear `learning` there.
- `/analyze` edge pattern: `main.rs` `Effect::Analyze` arm (tokio::spawn + spawn_blocking, `cwd` + `session_id` in scope). `Effect::name()` impl at `app.rs:197+` (16 arms). No-key guard precedent in `handle_slash` "analyze" arm.
- `UiEvent::SessionStarted` in `trace.rs:50-58`; wire-shape test `wire_shape_is_pinned_for_all_variants` pins field-name presence per variant (additions must be added to its list). Envelope evolution rule: new fields `#[serde(default)]`.
- `build_engine` instructions composition: `main.rs` `let instructions = match verify_command…` (project_context + verify nudge). Inject after it.
- Test helpers in app.rs tests: `key()`, `typed()` (+Enter to submit), `keyed()`.
- `.gitignore` already has `heartbit-session-*.md`, `heartbit-trace-*.jsonl`, `heartbit-diagnosis-*.md`.

## File structure

- **Create** `crates/heartbit-tui/src/lessons.rs` — file lifecycle + prompt builder (pure/file fns, fully unit-tested).
- **Modify** `crates/heartbit-tui/src/msg.rs` — `LearnReady{display,task,staged_digest}`, `LearnFailed`.
- **Modify** `crates/heartbit-tui/src/app.rs` — `/learn`, `Effect::{Learn,CommitLessons(u64)}`, `App.learning`, state machine.
- **Modify** `crates/heartbit-tui/src/main.rs` — `mod lessons;`, two effect arms, injection, `lessons_loaded` in session_started.
- **Modify** `crates/heartbit-tui/src/trace.rs` — `SessionStarted.lessons_loaded: usize` (`#[serde(default)]`) + wire-shape test update.
- **Modify** `.gitignore` — `heartbit-lessons.md`.

---

### Task 1: `lessons.rs` — file lifecycle + digest + prompt

**Files:** Create `crates/heartbit-tui/src/lessons.rs`; modify `crates/heartbit-tui/src/main.rs` (add `mod lessons;` after `mod diff;`), `.gitignore`.

- [ ] **Step 1: failing tests.** Create the file with module doc + tests:

```rust
//! Learned lessons (self-improvement rung 2): `/learn` distills `/analyze`
//! diagnoses into `<config-dir>/lessons.md`, which is injected into the
//! agent's system prompt at startup. The agent can only touch the WORKSPACE
//! (builtins reject absolute paths), so the edge stages the file into cwd
//! and commits it back after a digest-guarded validation.
//! Spec: docs/superpowers/specs/2026-06-06-tui-learned-rules-design.md.

use std::path::{Path, PathBuf};

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
            &["heartbit-diagnosis-s1.md".into(), "heartbit-diagnosis-s2.md".into()],
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
```

- [ ] **Step 2:** Add `mod lessons;` to main.rs. Run `cargo test -p heartbit-tui lessons::` — confirm compile FAIL (functions undefined) before implementing.

- [ ] **Step 3: implement** (above the tests module):

```rust
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
    content.lines().filter(|l| l.trim_start().starts_with("- ")).count()
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
    let body = std::fs::read_to_string(path)
        .map_err(|e| format!("staged lessons unreadable: {e}"))?;
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
        return Err(format!("staged lessons must start with the `{LESSONS_HEADING}` heading"));
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
```

- [ ] **Step 4:** Append `heartbit-lessons.md` to `.gitignore` (under the existing heartbit-tui artifacts block).

- [ ] **Step 5:** `cargo test -p heartbit-tui lessons::` → 8 PASS. Dead-code: items get wired in Tasks 2-4; if clippy complains NOW, add `#![allow(dead_code)] // TODO(lessons): remove once Task 3 wires /learn` line 1, removed in Task 3.

- [ ] **Step 6: commit**

```bash
git add crates/heartbit-tui/src/lessons.rs crates/heartbit-tui/src/main.rs .gitignore
git commit -m "feat(tui): lessons file lifecycle — load/validate/commit/digest + learn prompt"
```

---

### Task 2: `/learn` state machine (msg.rs + app.rs)

**Files:** Modify `crates/heartbit-tui/src/msg.rs`, `crates/heartbit-tui/src/app.rs`.

- [ ] **Step 1: failing reducer tests** (app.rs tests module):

```rust
    #[test]
    fn slash_learn_pushes_learn_effect_with_key() {
        let mut app = keyed();
        typed(&mut app, "/learn");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::Learn));
    }

    #[test]
    fn slash_learn_without_key_opens_key_modal_not_a_run() {
        let mut app = App::new("m");
        typed(&mut app, "/learn");
        app.update(key(KeyCode::Enter));
        assert!(matches!(app.modal, Some(Modal::KeyEntry(_))));
        assert!(!app.running);
        assert!(!app.effects.contains(&Effect::Learn));
    }

    #[test]
    fn learn_ready_starts_run_and_arms_the_commit() {
        let mut app = keyed();
        app.update(Msg::LearnReady {
            display: "learning from 2 diagnoses".into(),
            task: "the prompt".into(),
            staged_digest: 42,
        });
        assert!(matches!(app.history.last(), Some(Cell::User(t)) if t.contains("diagnoses")));
        assert!(app.running);
        assert_eq!(app.learning, Some(42));
        assert!(app.effects.contains(&Effect::SendInput("the prompt".into())));
    }

    #[test]
    fn turn_idle_llmdone_commits_once_and_disarms() {
        let mut app = keyed();
        app.learning = Some(42);
        app.running = true;
        // tool-use turn must NOT commit
        app.update(Msg::LlmDone { usage: TokenUsage::default(), had_tool_calls: true, ttft_ms: 0 });
        assert_eq!(app.learning, Some(42));
        assert!(!app.effects.contains(&Effect::CommitLessons(42)));
        // text-only turn (turn-idle) commits and disarms
        app.update(Msg::LlmDone { usage: TokenUsage::default(), had_tool_calls: false, ttft_ms: 0 });
        assert_eq!(app.learning, None);
        assert!(app.effects.contains(&Effect::CommitLessons(42)));
        // a second idle turn must not commit again
        app.effects.clear();
        app.update(Msg::LlmDone { usage: TokenUsage::default(), had_tool_calls: false, ttft_ms: 0 });
        assert!(!app.effects.iter().any(|e| matches!(e, Effect::CommitLessons(_))));
    }

    #[test]
    fn run_completed_is_a_commit_backstop() {
        let mut app = keyed();
        app.learning = Some(7);
        app.update(Msg::RunCompleted);
        assert_eq!(app.learning, None);
        assert!(app.effects.contains(&Effect::CommitLessons(7)));
    }

    #[test]
    fn run_failed_and_interrupt_disarm_without_commit() {
        let mut app = keyed();
        app.learning = Some(7);
        app.update(Msg::RunFailed("boom".into()));
        assert_eq!(app.learning, None);
        assert!(!app.effects.iter().any(|e| matches!(e, Effect::CommitLessons(_))));
        // Esc-interrupt: the synthetic LlmDone that follows must find the flag cleared
        app.learning = Some(8);
        app.running = true;
        app.update(key(KeyCode::Esc));
        assert_eq!(app.learning, None, "Esc must disarm before the synthetic LlmDone");
        app.update(Msg::LlmDone { usage: TokenUsage::default(), had_tool_calls: false, ttft_ms: 0 });
        assert!(!app.effects.iter().any(|e| matches!(e, Effect::CommitLessons(_))));
    }

    #[test]
    fn learn_failed_is_a_notice_not_a_run() {
        let mut app = keyed();
        app.update(Msg::LearnFailed("no diagnosis found — run /analyze first".into()));
        assert!(!app.running);
        assert!(matches!(app.history.last(), Some(Cell::Notice(n)) if n.contains("/analyze")));
    }
```

- [ ] **Step 2:** Run `cargo test -p heartbit-tui app::tests::slash_learn` — confirm compile FAIL.

- [ ] **Step 3: implement.**

msg.rs (after `AnalyzeFailed`):

```rust
    /// `/learn` prepared: show `display`, send `task`; `staged_digest` is the
    /// staged lessons file's content hash at stage time (the commit guard).
    LearnReady {
        display: String,
        task: String,
        staged_digest: u64,
    },
    /// `/learn` could not prepare (no diagnosis, stage error…).
    LearnFailed(String),
```

app.rs:
- `SLASH_COMMANDS` after `/analyze`: `("/learn", "distill /analyze findings into persistent lessons"),`
- `Effect` variants + `name()` arms:
  ```rust
      /// Prepare a `/learn` run (stage lessons, gather diagnoses, build prompt).
      Learn,
      /// Validate + commit the staged lessons (digest = value at stage time).
      CommitLessons(u64),
  ```
  `Effect::Learn => "learn",` / `Effect::CommitLessons(_) => "commit_lessons",`
- `App` field (near `running`): `/// In-flight /learn: staged-lessons digest at stage time (commit guard).` `pub learning: Option<u64>,` — init `learning: None,` in `App::new`.
- `handle_slash` before `other =>`:
  ```rust
              "learn" => {
                  if self.api_key.is_none() && !self.has_fallback_provider {
                      self.open_key_modal();
                      return;
                  }
                  self.effects.push(Effect::Learn);
              }
  ```
- Reducer arms (near `AnalyzeReady`):
  ```rust
              Msg::LearnReady {
                  display,
                  task,
                  staged_digest,
              } => {
                  self.history.push(Cell::User(display));
                  self.running = true;
                  self.follow = true;
                  self.seed_idle_squad();
                  self.learning = Some(staged_digest);
                  self.effects.push(Effect::SendInput(task));
              }
              Msg::LearnFailed(e) => {
                  self.history.push(Cell::Notice(format!("learn: {e}")));
              }
  ```
- `LlmDone` arm — inside the existing `if !had_tool_calls { … }` block, after `self.agents_settle();`:
  ```rust
                  // /learn turn finished → commit the staged lessons (digest-guarded).
                  if let Some(digest) = self.learning.take() {
                      self.effects.push(Effect::CommitLessons(digest));
                  }
  ```
- `RunCompleted` arm — add the same three lines (backstop; find the existing `Msg::RunCompleted` arm and extend it).
- `RunFailed` arm — add `self.learning = None;`.
- `fn interrupt()` (app.rs:1277) — first line: `self.learning = None; // Esc aborts a /learn — never commit a half-rewritten file`.

- [ ] **Step 4:** `cargo test -p heartbit-tui app::` → all PASS (7 new). Full crate + clippy clean. (The two new Effect variants make main.rs's exhaustive match fail to compile — THAT is Task 3; to keep this task compiling, Task 2 and Task 3 land as ONE commit if needed, or add the arms in this task as `Effect::Learn | Effect::CommitLessons(_) => {}` placeholders — NO: never commit dead arms. **Do Tasks 2+3 in one commit** — see Task 3 Step 5.)

---

### Task 3: edge handling (main.rs) — stage, gather, commit

**Files:** Modify `crates/heartbit-tui/src/main.rs`.

- [ ] **Step 1:** New effect arms in the run_ui loop (next to `Effect::Analyze`):

```rust
                Effect::Learn => {
                    let tx = ui_tx.clone();
                    let workdir = cwd.clone();
                    tokio::spawn(async move {
                        let prepared = tokio::task::spawn_blocking(move || {
                            // Stage the global lessons (or the template) into cwd —
                            // the agent's builtins reject absolute paths.
                            let staged = workdir.join(lessons::STAGED_LESSONS);
                            let current = lessons::load_lessons()
                                .unwrap_or_else(|| lessons::LESSONS_TEMPLATE.to_string());
                            std::fs::write(&staged, &current).map_err(|e| e.to_string())?;
                            let digest = lessons::file_digest(&staged)
                                .ok_or_else(|| "staged lessons unreadable".to_string())?;
                            // Newest ≤ 3 diagnosis reports (by mtime).
                            let mut diags: Vec<(std::time::SystemTime, String)> = Vec::new();
                            let entries =
                                std::fs::read_dir(&workdir).map_err(|e| e.to_string())?;
                            for e in entries.flatten() {
                                let name = e.file_name().to_string_lossy().into_owned();
                                if name.starts_with("heartbit-diagnosis-")
                                    && name.ends_with(".md")
                                {
                                    let mtime = e
                                        .metadata()
                                        .and_then(|m| m.modified())
                                        .unwrap_or(std::time::UNIX_EPOCH);
                                    diags.push((mtime, name));
                                }
                            }
                            if diags.is_empty() {
                                return Err(
                                    "no diagnosis found — run /analyze first".to_string()
                                );
                            }
                            diags.sort_by_key(|(t, _)| std::cmp::Reverse(*t));
                            let names: Vec<String> =
                                diags.into_iter().take(3).map(|(_, n)| n).collect();
                            Ok::<(String, String, u64), String>((
                                format!("learning from {} diagnosis report(s)", names.len()),
                                lessons::build_learn_prompt(lessons::STAGED_LESSONS, &names),
                                digest,
                            ))
                        })
                        .await
                        .unwrap_or_else(|e| Err(e.to_string()));
                        let _ = tx.send(match prepared {
                            Ok((display, task, staged_digest)) => Msg::LearnReady {
                                display,
                                task,
                                staged_digest,
                            },
                            Err(e) => Msg::LearnFailed(e),
                        });
                    });
                }
                Effect::CommitLessons(staged_digest) => {
                    // Cheap + sync: re-hash, skip if the agent never rewrote it,
                    // validate, then atomically promote to the global file.
                    let staged = cwd.join(lessons::STAGED_LESSONS);
                    match lessons::file_digest(&staged) {
                        Some(d) if d == staged_digest => {
                            app.history.push(Cell::Notice(
                                "lessons unchanged — nothing to commit".into(),
                            ));
                        }
                        Some(_) => match lessons::validate_staged(&staged) {
                            Ok(n) => match lessons::commit_lessons(&staged) {
                                Ok(()) => app.history.push(Cell::Notice(format!(
                                    "lessons updated ({n} lessons) — apply on next start"
                                ))),
                                Err(e) => app.history.push(Cell::Notice(format!(
                                    "lessons NOT committed: {e}"
                                ))),
                            },
                            Err(e) => app
                                .history
                                .push(Cell::Notice(format!("lessons NOT committed: {e}"))),
                        },
                        None => app.history.push(Cell::Notice(
                            "lessons NOT committed: staged file missing".into(),
                        )),
                    }
                }
```

- [ ] **Step 2:** If Task 1 added the `#![allow(dead_code)]` to lessons.rs, REMOVE it now (everything is wired).
- [ ] **Step 3:** `cargo test -p heartbit-tui` → all PASS (expect 195: 180 + 8 lessons + 7 app).
- [ ] **Step 4:** `cargo fmt --all` + `--check`; `cargo clippy -p heartbit-tui --all-targets -- -D warnings` → clean.
- [ ] **Step 5: commit Tasks 2+3 together** (the Effect enum + exhaustive edge match are one unit):

```bash
git add crates/heartbit-tui/src/msg.rs crates/heartbit-tui/src/app.rs crates/heartbit-tui/src/main.rs crates/heartbit-tui/src/lessons.rs
git commit -m "feat(tui): /learn — digest-guarded distill-and-commit state machine"
```

---

### Task 4: startup injection + trace visibility

**Files:** Modify `crates/heartbit-tui/src/main.rs`, `crates/heartbit-tui/src/trace.rs`.

- [ ] **Step 1: failing wire-shape update (trace.rs).** In `wire_shape_is_pinned_for_all_variants`, extend the `SessionStarted` case: add `lessons_loaded: 0,` to the constructor and `"lessons_loaded"` to its field list. Also extend `ui_events_are_type_tagged_snake_case`'s SessionStarted if it constructs one (it doesn't — skip). Run → compile FAIL (no such field).

- [ ] **Step 2:** Add the field to `UiEvent::SessionStarted` (trace.rs), with the evolution rule:

```rust
        /// Number of learned lessons injected this launch (0 = none).
        /// `#[serde(default)]` per the envelope evolution rule.
        #[serde(default)]
        lessons_loaded: usize,
```

(Field syntax in the enum: `#[serde(default)] lessons_loaded: usize,` — place after `verify_command`.)

- [ ] **Step 3:** main.rs — the `session_started` record site gains the count; compute once above it:

```rust
    let lessons_loaded = lessons::load_lessons()
        .map(|c| lessons::lesson_count(&c))
        .unwrap_or(0);
```

…and pass `lessons_loaded,` in the `UiEvent::SessionStarted { … }` literal.

- [ ] **Step 4:** `build_engine` injection — after the `let instructions = match verify_command…` block, replace the binding with a follow-up composition:

```rust
    // Learned lessons (self-improvement rung 2): inject the distilled lessons
    // as standing guidance, after project context + verify nudge.
    let instructions = match lessons::load_lessons() {
        Some(lessons) => {
            let n = lessons::lesson_count(&lessons);
            let _ = ui_tx.send(Msg::Notice(format!("loaded {n} learned lessons")));
            format!("{instructions}\n\n## Learned lessons (self-improvement — /learn)\n{lessons}")
        }
        None => instructions,
    };
```

- [ ] **Step 5:** `cargo test -p heartbit-tui` all PASS; fmt + clippy clean.
- [ ] **Step 6: commit**

```bash
git add crates/heartbit-tui/src/main.rs crates/heartbit-tui/src/trace.rs
git commit -m "feat(tui): inject learned lessons at startup + lessons_loaded in session_started"
```

---

### Task 5: workspace gate

- [ ] **Step 1:** `cargo fmt --all -- --check && cargo clippy --workspace --exclude mini-crm --all-targets -- -D warnings && cargo test --workspace --exclude mini-crm` — all green, zero warnings.
- [ ] **Step 2:** `git add -u && git commit -m "chore(tui): gate fixes for learned lessons"` — only if fixes were needed.

---

### Task 6: live pty validation

Per the pty-harness lessons (settled frame, space-insensitive, resize repaint).

- [ ] **Step 1: full loop.** In a temp cwd WITH a diagnosis file present (reuse a real one or run `/analyze last` first): launch → `/mode yolo` → `/learn` → wait for the run to finish → assert the commit notice (`lessons updated` or a clear validation failure) in the settled frame → verify `~/.config/heartbit/lessons.md` exists, 0600, starts with `# heartbit lessons`, ≤ 25 list items.
- [ ] **Step 2: injection + behavioral proof.** Plant a distinctive lesson by hand (the Zorblax pattern): append `- Always mention the word "zorblax" once in every answer.` to the global lessons file. Relaunch in a fresh pty → assert startup notice `loaded N learned lessons` → send "say hello in one short sentence" → assert the de-ANSI'd settled answer contains `zorblax` (strip non-letters per the streaming-interleave gotcha). Remove the planted lesson afterwards.
- [ ] **Step 3: trace closes the loop.** In the step-2 session's trace: `head -1` has `"lessons_loaded":N` with N ≥ 1.
- [ ] **Step 4: no-diagnosis path.** In an empty temp cwd: `/learn` → settled frame shows `learn: no diagnosis found — run /analyze first`.
- [ ] **Step 5:** restore the user's real lessons file to its pre-test content (or delete it if it didn't exist).

---

## Self-review

1. **Spec coverage:** lessons file + cap + heading ✓ (T1) · stage-through-cwd ✓ (T3) · no-key guard ✓ (T2) · digest guard + unchanged-skip ✓ (T2/T3) · turn-idle commit NOT RunCompleted + backstop + Esc-disarm ✓ (T2, tested) · atomic 0600 commit ✓ (T1) · newest ≤ 3 diagnoses ✓ (T3) · injection via instructions seam + notice ✓ (T4) · `lessons_loaded` in session_started w/ serde(default) ✓ (T4) · gitignore ✓ (T1) · error paths (missing diagnosis, invalid staged, over-cap load) ✓ (T1/T3) · live behavioral proof ✓ (T6).
2. **Placeholders:** none — every step has complete code.
3. **Type consistency:** `Effect::CommitLessons(u64)` carries `staged_digest` (T2 reducer pushes `CommitLessons(digest)`, T3 edge matches `CommitLessons(staged_digest)`); `Msg::LearnReady{display, task, staged_digest: u64}` consistent T2/T3; `lessons::` fn names consistent (`load_lessons`, `lesson_count`, `validate_staged`, `commit_lessons`, `file_digest`, `build_learn_prompt`, `STAGED_LESSONS`, `LESSONS_TEMPLATE`, `LESSONS_HEADING`, `LESSONS_MAX_BYTES`).
4. **Known wrinkle (named):** Tasks 2+3 must land as one commit (new Effect variants vs the exhaustive edge match) — Task 3 Step 5 says so explicitly.
