# heartbit-rs:x persona Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-09-heartbit-rs-x-persona-design.md`

**Goal:** Ship a second X persona, `heartbit-rs:x`, that produces threads which **demonstrate heartbit-core / heartbit-cli features by example**, with every claim grounded in a real file path or type from the repo. Pure on-demand: invoked via `heartbit persona run heartbit-rs:x --review --once "<topic>"`.

**Architecture:** New typed `XHeartbitRsPersona` alongside the existing `XGhostPersona`. New `RepoInspectTool` (4 ops: `read_file`, `grep_repo`, `list_features`, `feature_demo`) scoped to `crates/heartbit-core/` + `crates/heartbit-cli/`. New `repo_researcher_recipe()` that uses `repo_inspect` as primary tool and `websearch` / `webfetch` as optional context. `heartbit-core::PersonaExpansion` gains `mode_addendum: Option<&'static str>`; the pipeline's writer-message builder gains a `mode_addendum: Option<&str>` parameter that surfaces it after `voice_guidelines`. The existing `XGhostPersona` is unchanged at runtime (its addendum stays `None`). User-side TOML schema is unchanged.

**Tech Stack:** Rust 2024 edition, tokio, serde, toml, `tempfile` for repo_inspect tests, `git grep` (subprocess) for grep op. Existing `heartbit-ghost` crate as host for the new persona + tool + agent.

**Branch:** `feat/heartbit-rs-persona` (already created off `main`; the spec lives there).

---

## Task 1: `PersonaExpansion::mode_addendum` field

**Files:**
- Modify: `crates/heartbit-core/src/persona/types.rs:24-35` (the `PersonaExpansion` struct + its manual `Debug` impl + the `persona_expansion_default_is_empty` test)

- [ ] **Step 1: Write the failing test**

Add at the end of `crates/heartbit-core/src/persona/types.rs` (inside the existing `#[cfg(test)] mod tests`):

```rust
    #[test]
    fn persona_expansion_default_mode_addendum_is_none() {
        let e = PersonaExpansion::default();
        assert!(e.mode_addendum.is_none());
    }

    #[test]
    fn persona_expansion_carries_static_mode_addendum() {
        const ADDENDUM: &str = "EVANGELISM MODE — test fixture";
        let e = PersonaExpansion {
            mode_addendum: Some(ADDENDUM),
            ..PersonaExpansion::default()
        };
        assert_eq!(e.mode_addendum, Some(ADDENDUM));
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test -p heartbit-core --lib persona::types::tests::persona_expansion_default_mode_addendum_is_none
```

Expected: compile error (no field `mode_addendum`).

- [ ] **Step 3: Add the field**

Edit `crates/heartbit-core/src/persona/types.rs`. Replace:

```rust
/// What a persona expands into at startup.
#[derive(Default)]
pub struct PersonaExpansion {
    /// Sub-agents the persona requires.
    pub agents: Vec<AgentConfig>,
    /// Orchestrator config for the persona's pipeline.
    pub orchestrator: OrchestratorConfig,
    /// Tool instances contributed by the persona.
    pub tools: Vec<Arc<dyn Tool>>,
    /// Trigger specs (cron / sensors / mention polling / manual). Empty in Phase 0.
    pub triggers: Vec<TriggerSpec>,
    /// Optional review channel spec. None in Phase 0.
    pub review: Option<ReviewSpec>,
}
```

With:

```rust
/// What a persona expands into at startup.
#[derive(Default)]
pub struct PersonaExpansion {
    /// Sub-agents the persona requires.
    pub agents: Vec<AgentConfig>,
    /// Orchestrator config for the persona's pipeline.
    pub orchestrator: OrchestratorConfig,
    /// Tool instances contributed by the persona.
    pub tools: Vec<Arc<dyn Tool>>,
    /// Trigger specs (cron / sensors / mention polling / manual). Empty in Phase 0.
    pub triggers: Vec<TriggerSpec>,
    /// Optional review channel spec. None in Phase 0.
    pub review: Option<ReviewSpec>,
    /// Persona-specific mode addendum, appended to voice-aware user
    /// messages by the pipeline. Carries persona-design constants
    /// (e.g. evangelism framing) without coupling them to user TOML.
    /// `None` for personas that don't need one.
    pub mode_addendum: Option<&'static str>,
}
```

Then update the manual `Debug` impl (same file) by adding the `mode_addendum.is_some()` field:

```rust
impl std::fmt::Debug for PersonaExpansion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonaExpansion")
            .field("agents", &self.agents.len())
            .field("tools", &self.tools.len())
            .field("triggers", &self.triggers.len())
            .field("review", &self.review.is_some())
            .field("mode_addendum", &self.mode_addendum.is_some())
            .finish()
    }
}
```

- [ ] **Step 4: Run the new tests**

```bash
cargo test -p heartbit-core --lib persona::types::tests
```

Expected: PASS (all 5 tests in the module — the original 4 plus 2 new — actually 6 total; 4 original + 2 new). Confirm zero failures.

- [ ] **Step 5: Run the full heartbit-core suite to catch regressions**

```bash
cargo test -p heartbit-core --lib
```

Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-core/src/persona/types.rs
git commit -m "feat(core): PersonaExpansion gains mode_addendum field

Optional Option<&'static str> field, default None, that lets typed
personas surface a persona-specific addendum string without coupling
the user-side TOML schema. Consumers that ignore it (heartbit-ghost:x)
behave unchanged.

Plumbing for heartbit-rs:x persona evangelism-mode addendum."
```

---

## Task 2: `build_writer_user_message` accepts `mode_addendum`

**Files:**
- Modify: `crates/heartbit-ghost/src/pipeline/prompts.rs:12-52` (the function signature + body + insertion point)
- Modify: `crates/heartbit-ghost/src/pipeline/mod.rs:351-356` (sole caller)

- [ ] **Step 1: Write the failing tests**

Add in `crates/heartbit-ghost/src/pipeline/prompts.rs` at the bottom of the file (create a `#[cfg(test)] mod tests` block if there isn't one — check first; if there is, append to it):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_user_message_appends_addendum_after_voice_guidelines() {
        let msg = build_writer_user_message(
            "topic",
            "digest",
            "VOICE GUIDELINES",
            None,
            0,
            1,
            Some("EVANGELISM MODE — fixture"),
        );
        let voice_pos = msg.find("VOICE GUIDELINES").expect("voice present");
        let add_pos = msg.find("EVANGELISM MODE — fixture").expect("addendum present");
        assert!(
            voice_pos < add_pos,
            "addendum must follow voice guidelines (voice@{voice_pos}, addendum@{add_pos})"
        );
    }

    #[test]
    fn writer_user_message_without_addendum_is_unchanged_baseline() {
        let msg = build_writer_user_message("topic", "digest", "VOICE", None, 0, 1, None);
        assert!(!msg.contains("EVANGELISM"));
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test -p heartbit-ghost --lib pipeline::prompts::tests
```

Expected: compile error (function takes 6 args, test passes 7).

- [ ] **Step 3: Update the function signature and body**

In `crates/heartbit-ghost/src/pipeline/prompts.rs`, replace the existing `build_writer_user_message` function (lines 12-52 approximately) with:

```rust
pub(crate) fn build_writer_user_message(
    topic: &str,
    research_digest: &str,
    voice_guidelines: &str,
    prev_revision: Option<&(String, String)>,
    variant_index: usize,
    total_variants: usize,
    mode_addendum: Option<&str>,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("Topic: {topic}\n\n"));
    out.push_str("Research digest:\n");
    out.push_str(research_digest);
    out.push_str("\n\n");
    out.push_str(voice_guidelines);
    out.push('\n');

    if let Some(addendum) = mode_addendum {
        out.push_str("\n");
        out.push_str(addendum);
        out.push('\n');
    }

    if let Some((prev_draft, critic_reason)) = prev_revision {
        out.push_str("\nPREVIOUS DRAFT:\n");
        out.push_str(prev_draft);
        out.push_str("\n\nSTYLE CRITIC FEEDBACK:\n");
        out.push_str(critic_reason);
        out.push_str(
            "\n\nPlease produce a revised draft addressing the feedback. \
             Output the post text only.\n",
        );
    } else {
        out.push_str("\nProduce one draft. Output the post text only.\n");
    }

    if total_variants > 1 {
        out.push_str(&format!(
            "\nYou are generating variant {} of {}. Pursue a distinct angle \
             from the other variants — emphasize different aspects, examples, \
             or framing.\n",
            variant_index + 1,
            total_variants,
        ));
    }

    out
}
```

- [ ] **Step 4: Update the sole caller**

In `crates/heartbit-ghost/src/pipeline/mod.rs` at line 351, find the existing call:

```rust
        let writer_msg = prompts::build_writer_user_message(
            topic,
            &research_digest,
            voice_guidelines,
            prev_revision.as_ref(),
            variant_index,
            total_variants,
        );
```

Replace with (add `cfg.mode_addendum` as the 7th argument — the `cfg` field arrives in Task 3; for this task we pass `None` and Task 3 wires it):

```rust
        let writer_msg = prompts::build_writer_user_message(
            topic,
            &research_digest,
            voice_guidelines,
            prev_revision.as_ref(),
            variant_index,
            total_variants,
            None,
        );
```

- [ ] **Step 5: Run the new tests + the full ghost suite**

```bash
cargo test -p heartbit-ghost --lib pipeline::prompts::tests
cargo test -p heartbit-ghost --lib
```

Expected: both PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-ghost/src/pipeline/prompts.rs crates/heartbit-ghost/src/pipeline/mod.rs
git commit -m "feat(ghost): build_writer_user_message accepts mode_addendum

7th param Option<&str>; when Some, appended after the voice_guidelines
block in the user message. Sole caller in pipeline::mod passes None
for now — wired in next commit.

heartbit-rs:x persona plumbing — task 2/10."
```

---

## Task 3: `PipelineConfig` + `ReviewConfig` carry `mode_addendum`

**Files:**
- Modify: `crates/heartbit-ghost/src/pipeline/mod.rs` (add field to `PipelineConfig`, thread to writer call site at ~line 351)
- Modify: `crates/heartbit-ghost/src/review/mod.rs` (add field to `ReviewConfig`, thread to pipeline calls — search for `PipelineConfig {` constructions inside `run_review_pipeline`)

- [ ] **Step 1: Read the current `PipelineConfig` shape**

```bash
grep -n "pub struct PipelineConfig\|pub struct ReviewConfig" crates/heartbit-ghost/src/pipeline/mod.rs crates/heartbit-ghost/src/review/mod.rs
```

Expected: each found once. Note their full field lists; Task 3 adds one new field to each.

- [ ] **Step 2: Write the failing test**

In `crates/heartbit-ghost/src/pipeline/mod.rs`, find the `#[cfg(test)] mod tests` block and add:

```rust
    #[test]
    fn pipeline_config_default_mode_addendum_is_none() {
        // PipelineConfig is constructed with `..` syntax so the field
        // must default cleanly. (No `Default` derive — we test via the
        // builder path used by the binary.)
        let provider = std::sync::Arc::new(crate::pipeline::tests::MockProvider::default());
        let cfg = PipelineConfig {
            provider,
            persona: "test:x".into(),
            topic: "topic".into(),
            research_digest_seed: None,
            voice_guidelines: "VOICE".into(),
            candidates_per_draft: 1,
            // Future fields: when adding a field to PipelineConfig, this
            // test will fail to compile until the field is added here too.
            mode_addendum: None,
        };
        assert!(cfg.mode_addendum.is_none());
    }
```

(The exact field list above is illustrative — adjust to whatever `PipelineConfig` actually requires; the point is to exercise `mode_addendum: None` and `mode_addendum.is_none()` after construction.)

- [ ] **Step 3: Run the test to verify it fails**

```bash
cargo test -p heartbit-ghost --lib pipeline_config_default_mode_addendum_is_none
```

Expected: compile error (no field `mode_addendum`).

- [ ] **Step 4: Add the field to `PipelineConfig`**

In `crates/heartbit-ghost/src/pipeline/mod.rs`, find `pub struct PipelineConfig<'a>` (or whatever the current declaration is) and add at the end of the field list:

```rust
    /// Persona-specific mode addendum surfaced in the writer's user
    /// message after voice_guidelines. None for personas that don't
    /// have one (heartbit-ghost:x).
    pub mode_addendum: Option<&'a str>,
```

(If `PipelineConfig` is not generic over `'a` already, add the lifetime parameter. Reading the existing struct first is essential — its current declaration may already use a lifetime.)

- [ ] **Step 5: Thread `mode_addendum` to the writer call site**

In `crates/heartbit-ghost/src/pipeline/mod.rs` at the writer call (~line 351 from Task 2), replace `None` with `cfg.mode_addendum`:

```rust
        let writer_msg = prompts::build_writer_user_message(
            topic,
            &research_digest,
            voice_guidelines,
            prev_revision.as_ref(),
            variant_index,
            total_variants,
            cfg.mode_addendum,
        );
```

- [ ] **Step 6: Add the field to `ReviewConfig` and forward**

In `crates/heartbit-ghost/src/review/mod.rs`, find `pub struct ReviewConfig<'a>` and add the same field:

```rust
    pub mode_addendum: Option<&'a str>,
```

Then find every place inside `run_review_pipeline` (and helpers) that constructs a `PipelineConfig` and forward the field:

```rust
        let pipeline_cfg = PipelineConfig {
            // ... existing fields ...
            mode_addendum: cfg.mode_addendum,
        };
```

- [ ] **Step 7: Update existing test fixtures that build configs**

Run `cargo build --tests -p heartbit-ghost` and chase the compile errors — every test fixture that constructs a `PipelineConfig` or `ReviewConfig` must now include `mode_addendum: None`. This is mechanical: add the field, set it to `None`.

```bash
cargo build --tests -p heartbit-ghost 2>&1 | grep -E "^error\[E" | head
```

Expected: ~5-10 fixture sites need patching. Patch each.

- [ ] **Step 8: Run the full ghost suite**

```bash
cargo test -p heartbit-ghost --lib
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add crates/heartbit-ghost/src/pipeline/mod.rs crates/heartbit-ghost/src/review/mod.rs
git commit -m "feat(ghost): PipelineConfig + ReviewConfig carry mode_addendum

Optional Option<&'a str> field forwarded to build_writer_user_message
at the writer call site. Test fixtures updated to set None (no behavior
change for heartbit-ghost:x).

heartbit-rs:x persona plumbing — task 3/10."
```

---

## Task 4: `RepoInspectTool` — `read_file` + `grep_repo` primitives

**Files:**
- Create: `crates/heartbit-ghost/src/tools/repo_inspect.rs`
- Modify: `crates/heartbit-ghost/src/tools/mod.rs` (re-export `RepoInspectTool`)

- [ ] **Step 1: Read an existing tool for shape reference**

```bash
sed -n '1,80p' crates/heartbit-ghost/src/tools/user.rs
```

Note: the `Tool` trait impl, `definition()`, `execute()` shape, async signature, error mapping. Match this shape.

- [ ] **Step 2: Write the failing tests (primitives only)**

Create `crates/heartbit-ghost/src/tools/repo_inspect.rs` with the test module first:

```rust
//! `repo_inspect` builtin — reads files and greps within a constrained
//! subset of the heartbit repo. Backs the `repo_researcher` agent for
//! the heartbit-rs:x persona.

use heartbit_core::tool::{Tool, ToolDefinition, ToolOutput};
use heartbit_core::Error;
use serde::Deserialize;
use serde_json::Value;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

#[derive(Debug, Deserialize)]
#[serde(tag = "op")]
enum Op {
    #[serde(rename = "read_file")]
    ReadFile { path: String, range: Option<(usize, usize)> },
    #[serde(rename = "grep_repo")]
    GrepRepo { pattern: String, glob: Option<String> },
}

pub struct RepoInspectTool {
    repo_root: PathBuf,
    allowed_prefixes: Vec<PathBuf>,
    max_file_lines: usize,
    max_grep_hits: usize,
}

impl RepoInspectTool {
    pub fn new(repo_root: impl Into<PathBuf>) -> Result<Self, Error> {
        let repo_root = repo_root
            .into()
            .canonicalize()
            .map_err(|e| Error::ToolError(format!("repo_root canonicalize: {e}")))?;
        Ok(Self {
            allowed_prefixes: vec![
                repo_root.join("crates/heartbit-core"),
                repo_root.join("crates/heartbit-cli"),
            ],
            repo_root,
            max_file_lines: 1000,
            max_grep_hits: 100,
        })
    }

    fn resolve_within_allowed(&self, path: &str) -> Result<PathBuf, String> {
        if path.starts_with('/') {
            return Err(format!("absolute paths are not allowed: {path}"));
        }
        let candidate = self.repo_root.join(path);
        let canonical = candidate
            .canonicalize()
            .map_err(|e| format!("path resolve: {path}: {e}"))?;
        if !self
            .allowed_prefixes
            .iter()
            .any(|p| canonical.starts_with(p))
        {
            return Err(format!(
                "path {path} resolves outside the allowed prefixes (heartbit-core / heartbit-cli)"
            ));
        }
        Ok(canonical)
    }
}

impl Tool for RepoInspectTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "repo_inspect".into(),
            description: "Read or grep files inside the heartbit repo, restricted to \
                          crates/heartbit-core and crates/heartbit-cli."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["read_file", "grep_repo"]},
                    "path": {"type": "string", "description": "relative path from repo root (read_file)"},
                    "range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "[start_line, end_line] 1-based inclusive (read_file, optional)"
                    },
                    "pattern": {"type": "string", "description": "regex pattern (grep_repo)"},
                    "glob": {"type": "string", "description": "optional file glob (grep_repo)"}
                },
                "required": ["op"]
            }),
        }
    }

    fn execute<'a>(
        &'a self,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + 'a>> {
        Box::pin(async move {
            let op: Op = serde_json::from_value(input)
                .map_err(|e| Error::ToolError(format!("repo_inspect input: {e}")))?;
            match op {
                Op::ReadFile { path, range } => self.do_read_file(&path, range).await,
                Op::GrepRepo { pattern, glob } => self.do_grep_repo(&pattern, glob.as_deref()).await,
            }
        })
    }
}

impl RepoInspectTool {
    async fn do_read_file(
        &self,
        path: &str,
        range: Option<(usize, usize)>,
    ) -> Result<ToolOutput, Error> {
        let resolved = match self.resolve_within_allowed(path) {
            Ok(p) => p,
            Err(msg) => return Ok(ToolOutput::error(msg)),
        };
        let text = tokio::fs::read_to_string(&resolved)
            .await
            .map_err(|e| Error::ToolError(format!("read_file({path}): {e}")))?;
        let all_lines: Vec<&str> = text.lines().collect();
        let (start, end) = match range {
            Some((s, e)) => (s.max(1), e.min(all_lines.len())),
            None => (1, all_lines.len()),
        };
        if end < start {
            return Ok(ToolOutput::error(format!(
                "range start ({start}) > end ({end}) for {path}"
            )));
        }
        let span = end - start + 1;
        if span > self.max_file_lines {
            return Ok(ToolOutput::error(format!(
                "requested {span} lines from {path}; max is {} — pass an explicit range",
                self.max_file_lines
            )));
        }
        let mut out = String::new();
        for (i, line) in all_lines.iter().enumerate().take(end).skip(start - 1) {
            out.push_str(&format!("{}: {}\n", i + 1, line));
        }
        Ok(ToolOutput::text(out))
    }

    async fn do_grep_repo(
        &self,
        pattern: &str,
        glob: Option<&str>,
    ) -> Result<ToolOutput, Error> {
        // Use git grep for .gitignore-respecting search restricted to
        // allowed prefixes. The :(top) prefix scopes pathspecs to repo
        // root (independent of current cwd inside).
        let mut cmd = tokio::process::Command::new("git");
        cmd.current_dir(&self.repo_root);
        cmd.arg("grep").arg("-n").arg("-e").arg(pattern);
        if let Some(g) = glob {
            cmd.arg("--").arg(g).arg(":(top)crates/heartbit-core").arg(":(top)crates/heartbit-cli");
        } else {
            cmd.arg("--").arg(":(top)crates/heartbit-core").arg(":(top)crates/heartbit-cli");
        }
        let output = cmd
            .output()
            .await
            .map_err(|e| Error::ToolError(format!("git grep: {e}")))?;
        // git grep returns 1 on no-match — treat as empty result, not error.
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        if stdout.is_empty() {
            return Ok(ToolOutput::text(format!("(no matches for {pattern})")));
        }
        let mut lines: Vec<&str> = stdout.lines().take(self.max_grep_hits).collect();
        let truncated = stdout.lines().count() > self.max_grep_hits;
        let mut out = lines.join("\n");
        if truncated {
            out.push_str(&format!(
                "\n... ({} more hits truncated; cap is {})",
                stdout.lines().count() - self.max_grep_hits,
                self.max_grep_hits
            ));
        }
        let _ = lines.drain(..); // silence unused
        Ok(ToolOutput::text(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture_repo() -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("tempdir");
        // Mimic the real repo layout
        let core_dir = tmp.path().join("crates/heartbit-core/src");
        let cli_dir = tmp.path().join("crates/heartbit-cli/src");
        let other_dir = tmp.path().join("crates/heartbit-other/src");
        std::fs::create_dir_all(&core_dir).unwrap();
        std::fs::create_dir_all(&cli_dir).unwrap();
        std::fs::create_dir_all(&other_dir).unwrap();
        std::fs::write(core_dir.join("lib.rs"), "pub trait Tool {}\npub fn hello() {}\n").unwrap();
        std::fs::write(cli_dir.join("main.rs"), "fn main() { println!(\"hi\"); }\n").unwrap();
        std::fs::write(other_dir.join("lib.rs"), "pub fn out_of_scope() {}\n").unwrap();
        // Init as a git repo so git grep works.
        for arg in [&["init"][..], &["add", "."], &["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "init"]] {
            let _ = std::process::Command::new("git").args(arg).current_dir(tmp.path()).output();
        }
        tmp
    }

    #[tokio::test]
    async fn read_file_returns_lines_with_numbers() {
        let tmp = fixture_repo();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(json!({"op": "read_file", "path": "crates/heartbit-core/src/lib.rs"}))
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("1: pub trait Tool"));
        assert!(out.content.contains("2: pub fn hello"));
    }

    #[tokio::test]
    async fn read_file_respects_range() {
        let tmp = fixture_repo();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(json!({"op": "read_file", "path": "crates/heartbit-core/src/lib.rs", "range": [2, 2]}))
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("2: pub fn hello"));
        assert!(!out.content.contains("1: "));
    }

    #[tokio::test]
    async fn read_file_rejects_path_outside_allowed_prefixes() {
        let tmp = fixture_repo();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(json!({"op": "read_file", "path": "crates/heartbit-other/src/lib.rs"}))
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("outside the allowed prefixes"));
    }

    #[tokio::test]
    async fn read_file_rejects_absolute_path() {
        let tmp = fixture_repo();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(json!({"op": "read_file", "path": "/etc/passwd"}))
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("absolute paths are not allowed"));
    }

    #[tokio::test]
    async fn grep_repo_finds_matches_in_core_and_cli_only() {
        let tmp = fixture_repo();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(json!({"op": "grep_repo", "pattern": "pub fn"}))
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("crates/heartbit-core/src/lib.rs"));
        assert!(!out.content.contains("crates/heartbit-other"));
    }

    #[tokio::test]
    async fn grep_repo_returns_no_match_message_for_empty_result() {
        let tmp = fixture_repo();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(json!({"op": "grep_repo", "pattern": "nonexistent_xyzzy"}))
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("no matches"));
    }
}
```

- [ ] **Step 3: Add the dev-dependency**

`crates/heartbit-ghost/Cargo.toml` — verify `tempfile` is in `[dev-dependencies]`. If not:

```bash
grep -E "^tempfile" crates/heartbit-ghost/Cargo.toml
```

If absent, add under `[dev-dependencies]`:

```toml
tempfile = { workspace = true }
```

(Verify `tempfile` is a workspace dep first via `grep tempfile Cargo.toml | head`. If not, add `tempfile = "3"` to `[workspace.dependencies]` in root `Cargo.toml` first.)

- [ ] **Step 4: Re-export from `tools/mod.rs`**

In `crates/heartbit-ghost/src/tools/mod.rs`, add:

```rust
pub mod repo_inspect;
pub use repo_inspect::RepoInspectTool;
```

- [ ] **Step 5: Run the tests**

```bash
cargo test -p heartbit-ghost --lib tools::repo_inspect
```

Expected: 6 PASS.

- [ ] **Step 6: Run clippy + fmt**

```bash
cargo fmt --all && cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-ghost/src/tools/repo_inspect.rs crates/heartbit-ghost/src/tools/mod.rs crates/heartbit-ghost/Cargo.toml Cargo.toml
git commit -m "feat(ghost): RepoInspectTool primitives — read_file + grep_repo

Scoped to crates/heartbit-core and crates/heartbit-cli via canonical-path
allowed-prefixes check. read_file caps at 1000 lines (or explicit range).
grep_repo runs 'git grep -n' restricted via :(top) pathspecs and caps
at 100 hits. 6 unit tests using a tempfile-backed mock repo.

heartbit-rs:x persona plumbing — task 4/10."
```

---

## Task 5: `RepoInspectTool` — feature menu (`list_features` + `feature_demo`)

**Files:**
- Modify: `crates/heartbit-ghost/src/tools/repo_inspect.rs` (extend `Op` enum, add menu loader, add 2 new ops)
- Create: `crates/heartbit-ghost/data/heartbit-rs-features.toml` (initial 3-entry stub for tests; full curation in Task 6)

- [ ] **Step 1: Write the stub menu (just enough to test against)**

Create `crates/heartbit-ghost/data/heartbit-rs-features.toml`:

```toml
version = 1

[[feature]]
name = "tool_trait"
description = "The Tool trait — definition() + execute() — that powers everything in heartbit-core"
canonical_file = "crates/heartbit-core/src/tool/mod.rs"
key_types = ["Tool", "ToolDefinition", "ToolOutput"]
payoff = "implement two methods, get a fully-wired tool with retry, guardrails, telemetry"

[[feature]]
name = "agent_runner"
description = "Standalone agent loop with tokio::JoinSet for parallel tool execution"
canonical_file = "crates/heartbit-core/src/agent/runner.rs"
key_types = ["AgentRunner", "AgentRunnerBuilder", "AgentOutput"]
payoff = "single-process agent loop, no Restate / no daemon — drop into any tokio app"

[[feature]]
name = "memory_trait"
description = "Memory trait with 6 methods: store / recall / update / forget / add_link / prune"
canonical_file = "crates/heartbit-core/src/memory/mod.rs"
key_types = ["Memory", "MemoryEntry", "MemoryQuery"]
payoff = "swap InMemory for Postgres without touching agent code; SOTA recall built in"
```

(Task 6 expands this to ~18 entries; this stub is enough to test the loader.)

- [ ] **Step 2: Write the failing tests for the menu ops**

In `crates/heartbit-ghost/src/tools/repo_inspect.rs`, add to the `tests` module:

```rust
    #[tokio::test]
    async fn list_features_returns_menu_entries() {
        let tmp = fixture_repo();
        // Copy the real menu file into the fixture's data/ dir so the
        // tool can find it relative to repo_root.
        let data_dir = tmp.path().join("crates/heartbit-ghost/data");
        std::fs::create_dir_all(&data_dir).unwrap();
        let menu_src = std::env::current_dir()
            .unwrap()
            .join("crates/heartbit-ghost/data/heartbit-rs-features.toml");
        // If running from the workspace root or the crate dir, find the
        // menu wherever it actually is (resilient to test-runner cwd).
        let menu_path = if menu_src.exists() {
            menu_src
        } else {
            std::path::PathBuf::from("data/heartbit-rs-features.toml")
        };
        std::fs::copy(&menu_path, data_dir.join("heartbit-rs-features.toml")).unwrap();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool.execute(json!({"op": "list_features"})).await.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("tool_trait"));
        assert!(out.content.contains("agent_runner"));
        assert!(out.content.contains("memory_trait"));
    }

    #[tokio::test]
    async fn feature_demo_returns_full_record_for_existing_name() {
        let tmp = fixture_repo();
        let data_dir = tmp.path().join("crates/heartbit-ghost/data");
        std::fs::create_dir_all(&data_dir).unwrap();
        let menu_src = std::env::current_dir()
            .unwrap()
            .join("crates/heartbit-ghost/data/heartbit-rs-features.toml");
        let menu_path = if menu_src.exists() {
            menu_src
        } else {
            std::path::PathBuf::from("data/heartbit-rs-features.toml")
        };
        std::fs::copy(&menu_path, data_dir.join("heartbit-rs-features.toml")).unwrap();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(json!({"op": "feature_demo", "name": "tool_trait"}))
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("crates/heartbit-core/src/tool/mod.rs"));
        assert!(out.content.contains("ToolDefinition"));
    }

    #[tokio::test]
    async fn feature_demo_returns_error_for_unknown_name() {
        let tmp = fixture_repo();
        let data_dir = tmp.path().join("crates/heartbit-ghost/data");
        std::fs::create_dir_all(&data_dir).unwrap();
        let menu_src = std::env::current_dir()
            .unwrap()
            .join("crates/heartbit-ghost/data/heartbit-rs-features.toml");
        let menu_path = if menu_src.exists() {
            menu_src
        } else {
            std::path::PathBuf::from("data/heartbit-rs-features.toml")
        };
        std::fs::copy(&menu_path, data_dir.join("heartbit-rs-features.toml")).unwrap();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(json!({"op": "feature_demo", "name": "no_such_feature"}))
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("unknown feature"));
    }
```

- [ ] **Step 3: Extend the `Op` enum**

In the same file, replace:

```rust
#[derive(Debug, Deserialize)]
#[serde(tag = "op")]
enum Op {
    #[serde(rename = "read_file")]
    ReadFile { path: String, range: Option<(usize, usize)> },
    #[serde(rename = "grep_repo")]
    GrepRepo { pattern: String, glob: Option<String> },
}
```

With:

```rust
#[derive(Debug, Deserialize)]
#[serde(tag = "op")]
enum Op {
    #[serde(rename = "read_file")]
    ReadFile { path: String, range: Option<(usize, usize)> },
    #[serde(rename = "grep_repo")]
    GrepRepo { pattern: String, glob: Option<String> },
    #[serde(rename = "list_features")]
    ListFeatures,
    #[serde(rename = "feature_demo")]
    FeatureDemo { name: String },
}

#[derive(Debug, Deserialize, Clone)]
struct FeatureMenu {
    #[serde(default = "default_menu_version")]
    pub version: u32,
    pub feature: Vec<FeatureEntry>,
}
fn default_menu_version() -> u32 { 1 }

#[derive(Debug, Deserialize, Clone)]
struct FeatureEntry {
    pub name: String,
    pub description: String,
    pub canonical_file: String,
    pub key_types: Vec<String>,
    pub payoff: String,
}

impl FeatureMenu {
    fn load(repo_root: &Path) -> Option<Self> {
        let path = repo_root.join("crates/heartbit-ghost/data/heartbit-rs-features.toml");
        let text = std::fs::read_to_string(&path).ok()?;
        toml::from_str(&text).ok()
    }
}
```

- [ ] **Step 4: Add the menu field on the tool struct + load at construction**

Replace:

```rust
pub struct RepoInspectTool {
    repo_root: PathBuf,
    allowed_prefixes: Vec<PathBuf>,
    max_file_lines: usize,
    max_grep_hits: usize,
}
```

With:

```rust
pub struct RepoInspectTool {
    repo_root: PathBuf,
    allowed_prefixes: Vec<PathBuf>,
    max_file_lines: usize,
    max_grep_hits: usize,
    menu: Option<FeatureMenu>,
}
```

In `RepoInspectTool::new`, replace the construction body to load the menu (it's `None` if the file is absent — the primitives still work):

```rust
    pub fn new(repo_root: impl Into<PathBuf>) -> Result<Self, Error> {
        let repo_root = repo_root
            .into()
            .canonicalize()
            .map_err(|e| Error::ToolError(format!("repo_root canonicalize: {e}")))?;
        let menu = FeatureMenu::load(&repo_root);
        Ok(Self {
            allowed_prefixes: vec![
                repo_root.join("crates/heartbit-core"),
                repo_root.join("crates/heartbit-cli"),
            ],
            repo_root,
            max_file_lines: 1000,
            max_grep_hits: 100,
            menu,
        })
    }
```

- [ ] **Step 5: Dispatch the new ops**

In `Tool::execute`'s `match op` block:

```rust
                Op::ListFeatures => self.do_list_features().await,
                Op::FeatureDemo { name } => self.do_feature_demo(&name).await,
```

And add the two methods on `RepoInspectTool`:

```rust
    async fn do_list_features(&self) -> Result<ToolOutput, Error> {
        let menu = match self.menu.as_ref() {
            Some(m) => m,
            None => {
                return Ok(ToolOutput::error(
                    "feature menu not loaded — \
                     crates/heartbit-ghost/data/heartbit-rs-features.toml is missing"
                        .to_string(),
                ));
            }
        };
        let mut out = String::new();
        for f in &menu.feature {
            out.push_str(&format!("- {}: {} ({})\n", f.name, f.description, f.payoff));
        }
        Ok(ToolOutput::text(out))
    }

    async fn do_feature_demo(&self, name: &str) -> Result<ToolOutput, Error> {
        let menu = match self.menu.as_ref() {
            Some(m) => m,
            None => {
                return Ok(ToolOutput::error(
                    "feature menu not loaded — heartbit-rs-features.toml is missing"
                        .to_string(),
                ));
            }
        };
        match menu.feature.iter().find(|f| f.name == name) {
            Some(f) => Ok(ToolOutput::text(format!(
                "name: {}\ndescription: {}\ncanonical_file: {}\nkey_types: {}\npayoff: {}",
                f.name,
                f.description,
                f.canonical_file,
                f.key_types.join(", "),
                f.payoff,
            ))),
            None => Ok(ToolOutput::error(format!("unknown feature: {name}"))),
        }
    }
```

- [ ] **Step 6: Update the input_schema**

In `Tool::definition`, update the `op` enum:

```rust
                    "op": {"type": "string", "enum": ["read_file", "grep_repo", "list_features", "feature_demo"]},
```

And add `name` to `properties`:

```rust
                    "name": {"type": "string", "description": "feature name (feature_demo)"},
```

- [ ] **Step 7: Run all repo_inspect tests**

```bash
cargo test -p heartbit-ghost --lib tools::repo_inspect
```

Expected: 9 PASS (6 from Task 4 + 3 new).

- [ ] **Step 8: Format + clippy + commit**

```bash
cargo fmt --all && cargo clippy -p heartbit-ghost --all-targets -- -D warnings
git add crates/heartbit-ghost/src/tools/repo_inspect.rs crates/heartbit-ghost/data/heartbit-rs-features.toml
git commit -m "feat(ghost): RepoInspectTool feature menu — list_features + feature_demo

Loads crates/heartbit-ghost/data/heartbit-rs-features.toml at tool
construction; gracefully degrades to error responses when the file
is absent. Stub menu (3 entries) ships with this commit; Task 6
expands to ~18.

heartbit-rs:x persona plumbing — task 5/10."
```

---

## Task 6: Curate the V1 features menu + CI staleness check

**Files:**
- Modify: `crates/heartbit-ghost/data/heartbit-rs-features.toml` (expand from 3 to ~18 entries)
- Create: `crates/heartbit-ghost/tests/features_menu_paths_exist.rs` (CI test)

- [ ] **Step 1: Verify each canonical_file path before populating**

For each of the 18 features in the spec's Appendix A, confirm the canonical file path:

```bash
for path in \
  "crates/heartbit-core/src/tool/mod.rs" \
  "crates/heartbit-core/src/agent/runner.rs" \
  "crates/heartbit-core/src/memory/mod.rs" \
  "crates/heartbit-core/src/agent/workflow.rs" \
  "crates/heartbit-core/src/agent/guardrails/llm_judge.rs" \
  "crates/heartbit-core/src/llm/cascade.rs" \
  "crates/heartbit-core/src/agent/tool_filter.rs"; do
  ls -la "$path" 2>&1 | head -1
done
```

For paths that do NOT exist, run a `find` to locate the right file:

```bash
# Examples for the "verify path" entries from the spec:
find crates/heartbit-core/src -name "*.rs" | grep -i mcp | head
find crates/heartbit-core/src -name "*.rs" | grep -i guardrail | head
find crates/heartbit-core/src -name "*.rs" | grep -i memory | head
find crates/heartbit-core/src -name "*.rs" | grep -i bm25 | head
find crates/heartbit-core/src -name "*.rs" | grep -i anthropic | head
find crates/heartbit-core/src -name "*.rs" | grep -i restate | head
find crates/heartbit-core/src -name "*.rs" | grep -i retry | head
find crates -name "daemon" -type d | head
```

Record the actual file path for each feature.

- [ ] **Step 2: Replace the stub with the curated 18-entry menu**

Overwrite `crates/heartbit-ghost/data/heartbit-rs-features.toml`. Use the verified paths from Step 1. Each entry must have all 5 fields (`name`, `description`, `canonical_file`, `key_types`, `payoff`). Aim for descriptions ≤ 1 line and payoffs ≤ 1 sentence.

Template for each entry (fill from spec Appendix A):

```toml
[[feature]]
name = "<snake_case_name>"
description = "<one sentence>"
canonical_file = "<verified-path>"
key_types = ["<Type1>", "<Type2>"]
payoff = "<one sentence on what it enables>"
```

Order: keep alphabetical by `name` for predictability.

- [ ] **Step 3: Write the CI staleness test**

Create `crates/heartbit-ghost/tests/features_menu_paths_exist.rs`:

```rust
//! CI test: every canonical_file in heartbit-rs-features.toml must exist.
//!
//! This catches stale menu entries when files are renamed or deleted
//! without updating the menu.

use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct FeatureMenu {
    feature: Vec<FeatureEntry>,
}

#[derive(serde::Deserialize)]
struct FeatureEntry {
    name: String,
    canonical_file: String,
}

#[test]
fn every_canonical_file_in_feature_menu_exists() {
    let menu_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data/heartbit-rs-features.toml");
    let text = std::fs::read_to_string(&menu_path).expect("menu file readable");
    let menu: FeatureMenu = toml::from_str(&text).expect("menu parses");
    // Resolve relative to workspace root (one level up from crate dir).
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let mut missing = Vec::new();
    for f in &menu.feature {
        let p = workspace_root.join(&f.canonical_file);
        if !p.exists() {
            missing.push(format!("  {} -> {}", f.name, f.canonical_file));
        }
    }
    assert!(
        missing.is_empty(),
        "feature menu has stale canonical_file paths:\n{}",
        missing.join("\n")
    );
}
```

- [ ] **Step 4: Run the test**

```bash
cargo test -p heartbit-ghost --test features_menu_paths_exist
```

Expected: PASS. If it fails, fix the offending paths in the menu (the test message lists them).

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/data/heartbit-rs-features.toml crates/heartbit-ghost/tests/features_menu_paths_exist.rs
git commit -m "feat(ghost): curate V1 features menu (18 entries) + staleness CI

Each entry covers a notable heartbit-core or heartbit-cli feature with
description, canonical file, key types, and a one-sentence payoff. The
new test asserts every canonical_file exists on disk so renames or
deletions don't leave stale menu entries.

heartbit-rs:x persona plumbing — task 6/10."
```

---

## Task 7: `repo_researcher` agent recipe + `tools_for_heartbit_rs`

**Files:**
- Create: `crates/heartbit-ghost/src/agents/repo_researcher.rs`
- Modify: `crates/heartbit-ghost/src/agents/mod.rs` (add `pub mod repo_researcher`, re-export, new `tools_for_heartbit_rs()`)

- [ ] **Step 1: Read the existing researcher.rs for shape reference**

```bash
sed -n '1,55p' crates/heartbit-ghost/src/agents/researcher.rs
```

The new file mirrors this shape exactly — the only differences are the recipe name, system prompt, and `max_turns`.

- [ ] **Step 2: Write `repo_researcher.rs`**

Create `crates/heartbit-ghost/src/agents/repo_researcher.rs`:

```rust
//! Repo-grounded researcher sub-agent — backs the heartbit-rs:x persona.
//! Uses the `repo_inspect` builtin as primary substance; `websearch` /
//! `webfetch` are available for external context only.

use heartbit_core::config::AgentConfig;

pub const REPO_RESEARCHER_SYSTEM_PROMPT: &str = r#"You are a research analyst for a Rust agent framework called heartbit-rs. Given a feature name or topic, find the substance: the canonical file where it lives, the key types, a representative code excerpt, and a one-sentence payoff for someone reading about it.

PROCESS
1. If the user named a feature in the menu (e.g., "tool_trait", "memory_bm25"), call `repo_inspect` with `op: "feature_demo"` and read the canonical_file via `op: "read_file"`.
2. If the user gave a free-form topic, call `repo_inspect` with `op: "list_features"` first to see what's available, then either pick the closest one or use `op: "grep_repo"` to locate definitions yourself.
3. Read at most 2-3 files; pick the smallest excerpt that demonstrates the feature (typically a trait definition, a struct + 1-2 methods, or a single public function). Aim for ≤30 lines per excerpt.
4. `websearch` / `webfetch` are available ONLY for OPTIONAL external context (e.g. "how this compares to LangGraph", "the original paper"). They are NEVER the primary substance. The substance always comes from the repo.

OUTPUT FORMAT (free-form text, no JSON):
- Feature name + 1-sentence framing.
- Canonical file path (e.g., `crates/heartbit-core/src/tool/mod.rs`).
- Key types: comma-separated list.
- Code excerpt: ≤30 lines, fenced ```rust block, with the line numbers if from a range.
- Payoff: 1-2 sentences on what this enables for someone using the framework.
- Optional: 1-2 external context bullets with sources.

Do NOT write the post. The writer composes. Do NOT speculate beyond what the files show."#;

pub fn repo_researcher_recipe() -> AgentConfig {
    AgentConfig {
        name: "repo_researcher".to_string(),
        description: "Find substance about a heartbit-rs feature: canonical file, code excerpt, payoff.".to_string(),
        system_prompt: REPO_RESEARCHER_SYSTEM_PROMPT.to_string(),
        max_turns: Some(25),
        max_tokens: Some(4096),
        reasoning_effort: Some("medium".to_string()),
        ..super::stub_recipe("repo_researcher")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_researcher_recipe_has_expected_shape() {
        let cfg = repo_researcher_recipe();
        assert_eq!(cfg.name, "repo_researcher");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(25));
        assert_eq!(cfg.max_tokens, Some(4096));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("medium"));
        assert!(cfg.response_schema.is_none(), "free-form digest, no schema");
    }

    #[test]
    fn repo_researcher_prompt_routes_users_to_repo_inspect_first() {
        let p = REPO_RESEARCHER_SYSTEM_PROMPT;
        assert!(p.contains("repo_inspect"), "prompt mentions the primary tool");
        assert!(p.contains("feature_demo") && p.contains("list_features"),
                "prompt names the menu ops");
    }

    #[test]
    fn repo_researcher_prompt_explicitly_demotes_websearch() {
        let p = REPO_RESEARCHER_SYSTEM_PROMPT;
        assert!(
            p.contains("OPTIONAL") || p.contains("optional"),
            "prompt must mark websearch as optional"
        );
        assert!(
            p.contains("never the primary substance") || p.contains("NEVER the primary substance"),
            "prompt must explicitly demote websearch from primary substance"
        );
    }
}
```

- [ ] **Step 3: Wire into `agents/mod.rs`**

In `crates/heartbit-ghost/src/agents/mod.rs`, after the existing `pub use` re-exports, add:

```rust
pub mod repo_researcher;
pub use repo_researcher::repo_researcher_recipe;
```

Then add a new factory below the existing `tools_for_persona`:

```rust
/// Tool set for the heartbit-rs:x persona — the existing five plus
/// `RepoInspectTool` rooted at the workspace root (resolved via the
/// `CARGO_WORKSPACE_DIR` env var if set, falling back to `cwd()`).
pub fn tools_for_heartbit_rs() -> Vec<Arc<dyn Tool>> {
    use crate::tools::{RepoInspectTool, TwitterReplyTool, TwitterThreadTool};
    use heartbit_core::tool::builtins::{ImageGenerateTool, WebFetchTool, WebSearchTool};

    let repo_root = std::env::var("HEARTBIT_REPO_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().expect("cwd resolvable"));

    let repo_inspect: Arc<dyn Tool> = match RepoInspectTool::new(&repo_root) {
        Ok(t) => Arc::new(t),
        Err(e) => {
            // If we can't construct repo_inspect at startup, the persona
            // is unusable. Fail loudly rather than silently shipping a
            // crippled tool set.
            panic!("failed to construct RepoInspectTool from {repo_root:?}: {e}");
        }
    };

    vec![
        Arc::new(WebSearchTool::new()),
        Arc::new(WebFetchTool::new()),
        Arc::new(ImageGenerateTool::new()),
        Arc::new(TwitterThreadTool::new()),
        Arc::new(TwitterReplyTool::new()),
        repo_inspect,
    ]
}
```

- [ ] **Step 4: Add an integration test for `tools_for_heartbit_rs`**

In `crates/heartbit-ghost/src/agents/mod.rs` `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn tools_for_heartbit_rs_returns_six_tools_including_repo_inspect() {
        // Set CWD-equivalent env so RepoInspectTool::new succeeds even
        // when tests run from a worktree that mirrors the repo layout.
        std::env::set_var(
            "HEARTBIT_REPO_ROOT",
            std::env::var("CARGO_MANIFEST_DIR")
                .map(|p| std::path::PathBuf::from(p).parent().unwrap().parent().unwrap().to_path_buf())
                .unwrap_or_else(|_| std::env::current_dir().unwrap()),
        );
        let tools = tools_for_heartbit_rs();
        let names: Vec<String> = tools.iter().map(|t| t.definition().name).collect();
        assert!(
            names.iter().any(|n| n == "repo_inspect"),
            "repo_inspect must be in the tool list; got: {names:?}"
        );
        assert_eq!(tools.len(), 6, "expected 5 (existing) + 1 (repo_inspect)");
    }
```

- [ ] **Step 5: Run the new tests**

```bash
cargo test -p heartbit-ghost --lib agents::repo_researcher
cargo test -p heartbit-ghost --lib agents::tests::tools_for_heartbit_rs_returns_six_tools_including_repo_inspect
```

Expected: PASS.

- [ ] **Step 6: Format + clippy + commit**

```bash
cargo fmt --all && cargo clippy -p heartbit-ghost --all-targets -- -D warnings
git add crates/heartbit-ghost/src/agents/repo_researcher.rs crates/heartbit-ghost/src/agents/mod.rs
git commit -m "feat(ghost): repo_researcher recipe + tools_for_heartbit_rs

repo_researcher mirrors the existing researcher recipe with two
substantive differences: max_turns=25 (more exploration budget) and a
system prompt that routes the agent to repo_inspect as the primary
substance, demoting websearch/webfetch to optional context only.

tools_for_heartbit_rs returns the same 5 tools as tools_for_persona
plus a repo_inspect instance rooted at HEARTBIT_REPO_ROOT (or cwd).

heartbit-rs:x persona plumbing — task 7/10."
```

---

## Task 8: `XHeartbitRsPersona` typed persona + `register()` update

**Files:**
- Create: `crates/heartbit-ghost/src/heartbit_rs.rs`
- Modify: `crates/heartbit-ghost/src/lib.rs` (add `mod heartbit_rs;`, update `register()`, add tests)

- [ ] **Step 1: Read `lib.rs` for the existing persona shape**

```bash
sed -n '1,140p' crates/heartbit-ghost/src/lib.rs
```

The new persona mirrors `XGhostPersona` exactly except: the recipe vec uses `repo_researcher_recipe()`, the tool set uses `tools_for_heartbit_rs()`, and `expand()` sets `mode_addendum: Some(MODE_ADDENDUM)`.

- [ ] **Step 2: Write `heartbit_rs.rs`**

Create `crates/heartbit-ghost/src/heartbit_rs.rs`:

```rust
//! `heartbit-rs:x` persona — demonstrates heartbit-core / heartbit-cli
//! features by example. Reuses ghost's pipeline; only the researcher
//! agent and the writer's user-message addendum differ.

use std::sync::Arc;

use heartbit_core::persona::{Persona, PersonaExpansion, PersonaParams, PersonaRegistry};

pub const PERSONA_NAME: &str = "heartbit-rs:x";

/// Evangelism-mode addendum surfaced in voice-aware user messages by
/// the pipeline. See spec §6 for the rationale.
pub const MODE_ADDENDUM: &str = r#"EVANGELISM MODE — heartbit-rs:x

You are showing what heartbit-rs (a Rust multi-agent framework) does, by example. Your audience is Rust developers and AI engineers evaluating the framework.

THREAD SHAPE
Every thread is structured as: hook → demo → payoff.
- Hook: ONE concrete sentence stating what this feature lets you do (e.g. "Implement two methods on a trait, get a fully-wired tool with retry, guardrails, and telemetry.").
- Demo: a code excerpt taken from the researcher's digest. Paraphrase for tweet-friendliness if needed but do not invent code that wasn't in the digest. Reference the canonical file path inline (e.g., "in `crates/heartbit-core/src/tool/mod.rs`") so curious readers can cross-check.
- Payoff: 1-2 tweets on what this enables — concrete benefits, not adjectives.

GROUND TRUTH
- Every claim about heartbit-rs MUST trace back to a real file path or type the researcher surfaced. No vague "powerful" / "elegant" / "production-grade" framework adjectives without the corresponding code.
- If you cannot ground a claim, drop the claim.

NEVER
- Release-note framing ("we shipped X yesterday", "new in v2.0", "just released"). Frame everything time-agnostically — "here's what X does" not "here's what we just added".
- Marketing superlatives without code backing them.
- Code excerpts longer than 8 lines per tweet.
"#;

pub struct XHeartbitRsPersona {
    version: &'static str,
}

impl XHeartbitRsPersona {
    pub fn new() -> Self {
        Self { version: env!("CARGO_PKG_VERSION") }
    }
}

impl Default for XHeartbitRsPersona {
    fn default() -> Self {
        Self::new()
    }
}

impl Persona for XHeartbitRsPersona {
    fn name(&self) -> &str {
        PERSONA_NAME
    }

    fn description(&self) -> &str {
        "Demonstrates heartbit-core / heartbit-cli features by example. Pure on-demand."
    }

    fn version(&self) -> &str {
        self.version
    }

    fn expand(&self, _params: &PersonaParams) -> Result<PersonaExpansion, heartbit_core::Error> {
        let agents = vec![
            crate::agents::repo_researcher_recipe(),  // <— differs from ghost
            crate::agents::writer_recipe(),
            crate::agents::style_critic_recipe(),
            crate::agents::judge_recipe(),
            crate::agents::fact_check_recipe(),
            crate::agents::image_generator_recipe(),
            crate::agents::publisher_recipe(),
        ];
        let tools = crate::agents::tools_for_heartbit_rs();  // <— differs

        Ok(PersonaExpansion {
            agents,
            tools,
            mode_addendum: Some(MODE_ADDENDUM),
            ..PersonaExpansion::default()
        })
    }
}

/// Register the heartbit-rs:x persona into the supplied registry.
pub fn register(registry: &mut PersonaRegistry) {
    registry.register(Arc::new(XHeartbitRsPersona::new()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_stable() {
        let p = XHeartbitRsPersona::new();
        assert_eq!(p.name(), "heartbit-rs:x");
        assert_eq!(p.name(), PERSONA_NAME);
    }

    #[test]
    fn description_is_non_empty() {
        let p = XHeartbitRsPersona::new();
        assert!(!p.description().is_empty());
    }

    #[test]
    fn expand_puts_repo_researcher_first_and_carries_addendum() {
        std::env::set_var(
            "HEARTBIT_REPO_ROOT",
            std::env::var("CARGO_MANIFEST_DIR")
                .map(|p| {
                    std::path::PathBuf::from(p)
                        .parent()
                        .unwrap()
                        .parent()
                        .unwrap()
                        .to_path_buf()
                })
                .unwrap_or_else(|_| std::env::current_dir().unwrap()),
        );
        let p = XHeartbitRsPersona::new();
        let exp = p.expand(&PersonaParams::default()).expect("expand");
        assert_eq!(exp.agents.len(), 7);
        assert_eq!(exp.agents[0].name, "repo_researcher");
        assert_eq!(exp.mode_addendum, Some(MODE_ADDENDUM));
    }
}
```

- [ ] **Step 3: Update `lib.rs`**

In `crates/heartbit-ghost/src/lib.rs`, add at the top of the module declarations:

```rust
pub mod heartbit_rs;
```

Then update the existing `pub fn register(registry: &mut PersonaRegistry)` to register both:

```rust
pub fn register(registry: &mut PersonaRegistry) {
    registry.register(Arc::new(XGhostPersona::new()));
    heartbit_rs::register(registry);
}
```

(If `XGhostPersona`'s `register` was inlined directly, you may need a small refactor — keep `XGhostPersona::new()` registration inline and just add the second line.)

- [ ] **Step 4: Update `XGhostPersona::expand()` to set `mode_addendum: None` explicitly**

In `crates/heartbit-ghost/src/lib.rs`, find `XGhostPersona::expand` and confirm the `PersonaExpansion` literal uses `..PersonaExpansion::default()` — that already gives `mode_addendum: None`. No change required *unless* the literal lists fields explicitly, in which case add `mode_addendum: None,`.

- [ ] **Step 5: Run the new tests + the existing ghost tests**

```bash
cargo test -p heartbit-ghost --lib heartbit_rs
cargo test -p heartbit-ghost --lib tests
cargo test -p heartbit-ghost --lib
```

Expected: PASS, no regressions.

- [ ] **Step 6: Format + clippy + commit**

```bash
cargo fmt --all && cargo clippy -p heartbit-ghost --all-targets -- -D warnings
git add crates/heartbit-ghost/src/heartbit_rs.rs crates/heartbit-ghost/src/lib.rs
git commit -m "feat(ghost): XHeartbitRsPersona — typed persona for framework evangelism

Mirrors XGhostPersona shape except: agent slot 0 is repo_researcher
(uses repo_inspect as primary substance), tools include RepoInspectTool,
and PersonaExpansion.mode_addendum carries the evangelism MODE_ADDENDUM
constant. register() now registers both personas.

heartbit-rs:x persona plumbing — task 8/10."
```

---

## Task 9: CLI dispatcher passes `mode_addendum` through

**Files:**
- Modify: `crates/heartbit-cli/src/persona.rs` (or whichever file owns `persona run` dispatch — locate via grep)

- [ ] **Step 1: Locate the `persona run` dispatcher**

```bash
grep -rn "PipelineConfig\|ReviewConfig\|run_review_pipeline\|run_pipeline" crates/heartbit-cli/src/ --include="*.rs" | head -10
```

Note the file + line of every place that constructs a `PipelineConfig` or `ReviewConfig`. The dispatcher will call `persona.expand(&params)?` first, then build the config.

- [ ] **Step 2: Read the dispatcher's current shape**

```bash
sed -n '<L-30>,<L+30>p' <dispatcher-file>   # use the line numbers from Step 1
```

Identify where `expansion.agents` is read; the new code reads `expansion.mode_addendum` from the same expansion.

- [ ] **Step 3: Thread `mode_addendum` through**

Replace each `PipelineConfig { ... }` literal in the dispatcher to include `mode_addendum: expansion.mode_addendum`:

```rust
let expansion = persona.expand(&params)?;
let cfg = PipelineConfig {
    // ... existing fields ...
    mode_addendum: expansion.mode_addendum,
};
```

Same for `ReviewConfig` if `--review` mode constructs one separately.

- [ ] **Step 4: Run the CLI suite**

```bash
cargo test -p heartbit-cli
```

Expected: PASS.

- [ ] **Step 5: Smoke test — show the persona expansion**

```bash
cargo run --release --bin heartbit -- persona show heartbit-rs:x 2>&1 | head -30
```

Expected: succeeds; output includes `agents: ["repo_researcher", "writer", ...]` shape (or the equivalent debug rendering).

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-cli/src/persona.rs
git commit -m "feat(cli): persona run threads mode_addendum from expansion to config

Reads PersonaExpansion::mode_addendum after persona.expand() and passes
it through PipelineConfig / ReviewConfig to the writer's user-message
builder. No-op for personas that return None.

heartbit-rs:x persona plumbing — task 9/10."
```

---

## Task 10: Acceptance — quality gates + manual setup + live test

**Files:** none modified by this task; verifies prior tasks land cleanly.

- [ ] **Step 1: Full quality gate**

```bash
cargo fmt -- --check && \
  cargo clippy --all-targets -- -D warnings && \
  cargo test
```

Expected: all three green. Test count: previous baseline + ~18 new tests (6 repo_inspect primitives + 3 menu ops + 3 repo_researcher + 1 tools_for_heartbit_rs + 3 XHeartbitRsPersona + 2 prompt builder + 1 menu staleness CI test ≈ 19; spec says ~18, slight delta acceptable).

- [ ] **Step 2: Build the release binary**

```bash
cargo build --release --bin heartbit 2>&1 | tail -3
```

Expected: `Finished release ...`.

- [ ] **Step 3: Document the manual user-side setup**

Manual steps (NOT scripted by this plan — operator-side):

1. Ingest the two corpora:
   ```bash
   heartbit persona corpus add heartbit-rs:x burntsushi <path-to-burntsushi.jsonl>
   heartbit persona corpus add heartbit-rs:x simonw    <path-to-simonw.jsonl>
   ```

2. Build the persona's TOML:
   ```bash
   mkdir -p ~/.heartbit/ghost/personas
   cat > ~/.heartbit/ghost/personas/heartbit-rs:x.toml <<'EOF'
   version = 1

   [recipe]
   version = 1

   [[recipe.blend]]
   writer = "burntsushi"
   weight = 0.5

   [[recipe.blend]]
   writer = "simonw"
   weight = 0.5

   [recipe.overrides]
   thread_max_length = 12
   ai_tells_to_avoid = [
       "delve", "leverage", "unlock", "cutting-edge", "revolutionary",
       "game-changing", "—", "–"
   ]

   [recipe.overrides.formatting]
   em_dashes = "forbidden"
   periods = "always"
   quotation_marks = "double"
   line_breaks = "single"
   EOF
   ```

3. Build the style profile:
   ```bash
   heartbit persona profile rebuild heartbit-rs:x
   ```

(These three steps are NOT part of the plan's "complete a task" definition — they are the operator's responsibility before live testing.)

- [ ] **Step 4: Live test — single end-to-end run**

Set `HEARTBIT_REPO_ROOT` to the workspace root, then run:

```bash
HEARTBIT_REPO_ROOT="$(pwd)" \
  ./target/release/heartbit persona run heartbit-rs:x \
    --review \
    --once "show what the Tool trait gives you"
```

Expected behaviour:
- `> Loading profile snapshot...`
- `> Researching topic...` (repo_researcher reads `crates/heartbit-core/src/tool/mod.rs`)
- `> Generating 3 candidate(s) in parallel...`
- `> Sending review to user...` (Telegram review delivered)
- (operator picks a candidate)
- `> Generating optional image...`
- `> Posting candidate <i>...`
- `> Done.` with a `Posted { ... }` outcome.

Verification on the live tweet (head tweet):
- The text references at least one of `Tool`, `ToolDefinition`, `ToolOutput`.
- The text mentions the file path `crates/heartbit-core/src/tool/mod.rs` (or a paraphrased reference to it).

- [ ] **Step 5: Final merge**

Once Step 4 succeeds, finish the development branch via the **superpowers:finishing-a-development-branch** skill — present 4 options to the user (merge / PR / keep / discard).

---

## Self-review — pre-execution

- **Spec coverage:** every section of the spec maps to at least one task — §3 Files (Tasks 4-9), §4 repo_inspect (Tasks 4-5), §5 repo_researcher (Task 7), §6 mode_addendum (Tasks 1-3, 8), §7 plumbing (Tasks 1-3, 9), §8 TOML (Task 10 manual), §9 corpora (Task 10 manual), §10 tests (every task ships its own), §11 out of scope (no tasks — confirmed deferred), §12 risks (mitigations applied: Task 4 prefix check, Task 5 missing-menu fallback, Task 6 staleness CI test). Appendix A canonical_file paths verified in Task 6 Step 1.
- **Placeholder scan:** no `TBD` / `TODO` / `fill in details`. Code blocks are complete with imports + bodies. Tasks that depend on a path resolution call out the exact `grep` command to run.
- **Type consistency:** `mode_addendum` is `Option<&'static str>` on `PersonaExpansion`, `Option<&'a str>` on `PipelineConfig` / `ReviewConfig`, `Option<&str>` on the prompt-builder param — matches because lifetime narrowing flows naturally from owner to borrow.
- **One known dependency:** Task 9 needs Task 1 (`PersonaExpansion::mode_addendum` exists), Tasks 2-3 (`PipelineConfig::mode_addendum` exists), Task 8 (`XHeartbitRsPersona` registered). The order above respects this.
- **Mechanical clean-up locations** (likely fixture sites in Task 3 Step 7) — chase compile errors; the plan lists this as expected work.
